from pathlib import Path
from pydub import AudioSegment
import os, subprocess
import ffmpeg


FFBIN = ffmpeg.FFBIN_path()

def convert(mp3_path, wav_path, bin_dir=FFBIN):
    """
    Convert MP3 to WAV using ffmpeg command line.
    """
    mp3, wav = Path(mp3_path).resolve(), Path(wav_path).resolve()
    assert mp3.exists(), f"Missing input: {mp3}"
    wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(bin_dir/"ffmpeg.exe"), "-y", "-i", str(mp3), str(wav)]
    subprocess.run(cmd, check=True)
    return AudioSegment.from_wav(str(wav))

# convert(r"notwav_audio\Audio3.mp3", r"wav_audio\audio3.wav")
