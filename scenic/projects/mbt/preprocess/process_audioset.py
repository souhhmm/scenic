# scenic/projects/mbt/preprocess/process_audioset.py

import os
import math
import logging
import argparse
from pathlib import Path
import multiprocessing as mp
from typing import Tuple, List, Optional

import numpy as np
import tensorflow as tf
import librosa
import cv2
import yt_dlp
from tqdm import tqdm
import csv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('audioset_processing.log'),
        logging.StreamHandler()
    ]
)

class AudioSetProcessor:
    def __init__(self, 
                 base_dir: str,
                 num_workers: int = 4,
                 shard_size: int = 1024):
        """
        Initialize AudioSet processor.
        
        Args:
            base_dir: Base directory containing CSV files and for output
            num_workers: Number of parallel workers
            shard_size: Number of examples per TFRecord shard
        """
        self.base_dir = Path(base_dir)
        self.num_workers = num_workers
        self.shard_size = shard_size
        
        # Create output directories with parents=True
        self.tfrecord_dir = self.base_dir / 'tfrecords'
        self.tfrecord_dir.mkdir(parents=True, exist_ok=True)
        
        # Temp directory for downloads
        self.temp_dir = self.base_dir / 'temp'
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def extract_mel_spectrogram(self, 
                              audio: np.ndarray, 
                              sr: int = 16000) -> np.ndarray:
        """Extract mel spectrogram according to paper specifications."""
        # 25ms window = 400 samples at 16kHz
        # 10ms hop = 160 samples at 16kHz
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=400,
            hop_length=160,
            n_mels=128,
            window='hamming'
        )
        
        log_mel_spec = librosa.power_to_db(mel_spec)
        
        # Ensure correct time dimension (100 bins per second)
        expected_time_bins = int(100 * (len(audio) / sr))
        log_mel_spec = cv2.resize(log_mel_spec, (expected_time_bins, 128))
        
        return log_mel_spec

    def extract_video_frames(self,
                           video_path: str,
                           num_frames: int = 8,
                           size: Tuple[int, int] = (224, 224)) -> Optional[List[np.ndarray]]:
        """Extract frames according to paper specifications."""
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                return None
                
            # Sample 8 frames uniformly
            frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
            
            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, size)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
            
            cap.release()
            return frames if len(frames) == num_frames else None
            
        except Exception as e:
            logging.error(f"Error extracting frames: {str(e)}")
            return None

    def create_patches(self, image: np.ndarray, patch_size: int = 16) -> np.ndarray:
        """Create 16x16 patches from image."""
        h, w = image.shape[:2]
        patches = []
        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                if i + patch_size <= h and j + patch_size <= w:
                    patch = image[i:i+patch_size, j:j+patch_size]
                    patches.append(patch)
        return np.array(patches)

    def process_single_video(self, 
                        youtube_id: str, 
                        start_time: float, 
                        duration: float = 8.0) -> Tuple[Optional[List[np.ndarray]], 
                                                      Optional[np.ndarray]]:
        """Process a single video from YouTube."""
        try:
            # Use yt-dlp with modified options
            ydl_opts = {
                'format': 'bestaudio[ext=m4a]/bestaudio',  # First try to get m4a audio, then any audio
                'outtmpl': str(self.temp_dir / '%(id)s.%(ext)s'),
                'quiet': False,
                'no_warnings': False,
                'extract_audio': True,  # Extract audio
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'm4a',
                }]
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                try:
                    info = ydl.extract_info(f'https://youtube.com/watch?v={youtube_id}', download=True)
                    logging.info(f"Successfully downloaded audio for {youtube_id}")
                    
                    # Get the audio file path
                    temp_audio = self.temp_dir / f'{youtube_id}.m4a'
                    
                    if temp_audio.exists():
                        logging.info(f"Processing audio for {youtube_id}")
                        audio, sr = librosa.load(str(temp_audio), 
                                               sr=16000, 
                                               offset=start_time, 
                                               duration=duration)
                        spectrogram = self.extract_mel_spectrogram(audio)
                        logging.info(f"Audio processing complete for {youtube_id}")
                    else:
                        logging.error(f"Audio file not found for {youtube_id}")
                        return None, None

                    # For video frames, download video separately
                    video_opts = {
                        'format': 'bestvideo[ext=mp4]',
                        'outtmpl': str(self.temp_dir / '%(id)s_video.%(ext)s'),
                        'quiet': True
                    }
                    
                    with yt_dlp.YoutubeDL(video_opts) as video_dl:
                        video_dl.download([f'https://youtube.com/watch?v={youtube_id}'])
                        temp_video = self.temp_dir / f'{youtube_id}_video.mp4'
                        
                        if temp_video.exists():
                            frames = self.extract_video_frames(str(temp_video))
                            if frames is None:
                                logging.error(f"Frame extraction failed for {youtube_id}")
                                return None, None
                            logging.info(f"Video processing complete for {youtube_id}")
                        else:
                            logging.error(f"Video file not found for {youtube_id}")
                            return None, None

                    # Cleanup
                    temp_video.unlink(missing_ok=True)
                    temp_audio.unlink(missing_ok=True)
                    
                    return frames, spectrogram
                    
                except Exception as e:
                    logging.error(f"Download failed for {youtube_id}: {str(e)}")
                    return None, None

        except Exception as e:
            logging.error(f"Error processing {youtube_id}: {str(e)}")
            # Cleanup on error
            temp_video = self.temp_dir / f'{youtube_id}_video.mp4'
            temp_audio = self.temp_dir / f'{youtube_id}.m4a'
            temp_video.unlink(missing_ok=True)
            temp_audio.unlink(missing_ok=True)
            return None, None

    def create_tfrecord_example(self,
                               video_frames: List[np.ndarray],
                               spectrogram: np.ndarray,
                               label: List[int]) -> tf.train.SequenceExample:
        """Create TFRecord example with both modalities."""
        # Create patches
        video_patches = [self.create_patches(frame) for frame in video_frames]
        spec_patches = self.create_patches(spectrogram)
        
        context = tf.train.Features(
            feature={
                'label': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=label)
                )
            }
        )
        
        feature_lists = tf.train.FeatureLists(
            feature_list={
                'rgb': tf.train.FeatureList(
                    feature=[
                        tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[patches.tobytes()]
                            )
                        ) for patches in video_patches
                    ]
                ),
                'melspec/feature/floats': tf.train.FeatureList(
                    feature=[
                        tf.train.Feature(
                            float_list=tf.train.FloatList(
                                value=spec_patches.flatten().tolist()
                            )
                        )
                    ]
                )
            }
        )
        
        return tf.train.SequenceExample(context=context, feature_lists=feature_lists)

    def process_shard(self, shard_data):
        """Process a single shard of data."""
        shard_id, examples, output_path = shard_data
        successful_writes = 0
        
        writer = tf.io.TFRecordWriter(output_path)
        
        for youtube_id, start_time, labels in examples:
            frames, spectrogram = self.process_single_video(youtube_id, start_time)
            if frames is not None and spectrogram is not None:
                try:
                    example = self.create_tfrecord_example(frames, spectrogram, labels)
                    writer.write(example.SerializeToString())
                    successful_writes += 1
                    logging.info(f"Successfully processed video {youtube_id}")
                except Exception as e:
                    logging.error(f"Failed to create TFRecord for {youtube_id}: {str(e)}")
        
        writer.close()
        logging.info(f"Shard {shard_id} completed. Successfully wrote {successful_writes} examples")

    def process_dataset(self, csv_path: str, output_prefix: str):
        """Process entire dataset with sharding."""
        # Read and parse CSV
        examples = []
        with open(csv_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                youtube_id, start_time, _, labels = line.strip().split(', ')
                label_ids = [int(l) for l in labels.split('"')[1].split(',')]
                examples.append((youtube_id, float(start_time), label_ids))
        
        # Create shards
        num_shards = math.ceil(len(examples) / self.shard_size)
        shard_data = []
        
        for shard_id in range(num_shards):
            start_idx = shard_id * self.shard_size
            end_idx = min((shard_id + 1) * self.shard_size, len(examples))
            shard_examples = examples[start_idx:end_idx]
            
            if num_shards == 1:
                shard_filename = f"{output_prefix}.tfrecord"
            else:
                shard_filename = f"{output_prefix}.tfrecord-{shard_id:05d}-of-{num_shards:05d}"
            
            output_path = self.tfrecord_dir / shard_filename
            shard_data.append((shard_id, shard_examples, str(output_path)))
        
        # Process shards in parallel
        with mp.Pool(self.num_workers) as pool:
            list(tqdm(pool.imap(self.process_shard, shard_data),
                     total=len(shard_data),
                     desc="Processing shards"))

    def process_dataset_test(self, csv_path: str, output_prefix: str, num_videos: int = 10):
        """Process limited number of videos for testing."""
        # First, read the labels file to create mapping
        label_map = {}
        with open(os.path.join(self.base_dir, 'class_labels_indices.csv'), 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                index, mid, _ = row
                label_map[mid] = int(index)
        
        # Read and parse CSV
        examples = []
        with open(csv_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                youtube_id, start_time, _, labels = line.strip().split(', ')
                # Convert MID strings to integers using the mapping
                label_strings = labels.strip('"').split(',')
                label_ids = [label_map[label] for label in label_strings]
                examples.append((youtube_id, float(start_time), label_ids))
                if len(examples) >= num_videos:
                    break
        
        # Create single shard for test
        output_path = self.tfrecord_dir / f"{output_prefix}_test.tfrecord"
        shard_data = [(0, examples, str(output_path))]
        
        # Process videos
        logging.info(f"Processing {num_videos} videos for testing...")
        self.process_shard(shard_data[0])
        logging.info("Test processing completed!")

def main():
    parser = argparse.ArgumentParser(description='Process AudioSet for MBT')
    parser.add_argument('--base_dir', type=str, required=True,
                       help='Base directory containing CSV files and for output')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of parallel workers')
    parser.add_argument('--shard_size', type=int, default=1024,
                       help='Number of examples per shard')
    parser.add_argument('--test_mode', action='store_true',
                       help='Process only first 10 videos for testing')
    parser.add_argument('--num_test_videos', type=int, default=20,
                       help='Number of videos to process in test mode')
    
    args = parser.parse_args()
    
    processor = AudioSetProcessor(
        base_dir=args.base_dir,
        num_workers=args.num_workers,
        shard_size=args.shard_size
    )
    
    if args.test_mode:
        # Process limited videos for testing
        processor.process_dataset_test(
            csv_path=os.path.join(args.base_dir, 'balanced_train_segments.csv'),
            output_prefix='balanced_train.se.melspec',
            num_videos=args.num_test_videos
        )
    else:
        # Process full dataset
        processor.process_dataset(
            csv_path=os.path.join(args.base_dir, 'balanced_train_segments.csv'),
            output_prefix='balanced_train.se.melspec'
        )
        
        processor.process_dataset(
            csv_path=os.path.join(args.base_dir, 'eval_segments.csv'),
            output_prefix='eval.se.melspec'
        )

if __name__ == "__main__":
    main()