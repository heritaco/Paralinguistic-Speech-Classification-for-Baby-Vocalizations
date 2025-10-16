import os
from pydub import AudioSegment
import pathlib as pl

from scripts import ffmpeg
from scripts import mp3_to_wav


# Create function to convert audio file to wav
def convert_to_wav(mp3_path, wav_path):
  """Takes an audio file of non .wav format and converts to .wav"""
  FFBIN = ffmpeg.FFBIN_path()
  mp3_to_wav.convert(mp3_path, wav_path, bin_dir=FFBIN)

def show_pydub_stats(filename):
  """Returns different audio attributes related to an audio file."""
  # Create AudioSegment instance
  audio_segment = AudioSegment.from_file(filename)

  # Print audio attributes and return AudioSegment instance
  print(f"Channels: {  audio_segment.channels}")
  print(f"Sample width: {audio_segment.sample_width}")
  print(f"Frame rate (sample rate): {audio_segment.frame_rate}")
  print(f"Frame width: {audio_segment.frame_width}")
  print(f"Length (ms): {len(audio_segment)}")
  return audio_segment


def transcribe_audio(filename):
  """Takes a .wav format audio file and transcribes it to text."""

  import speech_recognition as sr
  
  # Setup a recognizer instance
  recognizer = sr.Recognizer()
  
  # Import the audio file and convert to audio data
  audio_file = sr.AudioFile(filename)
  with audio_file as source:
    audio_data = recognizer.record(source)
  
  # Return the transcribed text
  return recognizer.recognize_google(audio_data)



def convert_folder_to_wav(in_dir, out_dir):
    """Converts all .mp3 files in a folder to .wav format."""
    in_dir, out_dir = pl.Path(in_dir), pl.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outs = []
    for p in in_dir.glob("*.mp3"):
        out = convert_to_wav(p, out_dir)  # canonicalizes to mono/16k/16-bit WAV
        outs.append(out)
    return outs

def create_text_list(wav_dir):
    """
    Transcribes all .wav files in a folder to text and returns a list of texts.
    """
    wav_dir = pl.Path(wav_dir)
    texts = []
    for p in wav_dir.glob("*.wav"):
        txt = transcribe_audio(p)
        texts.append(txt)
    return texts