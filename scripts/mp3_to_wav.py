from pathlib import Path
from pydub import AudioSegment
import os, subprocess

def find_ffbin(start: Path) -> Path:
    # 1) search recursively under current tree
    cand = next(start.rglob("bin/ffmpeg.exe"), None)
    if cand: return cand.parent
    # 2) walk up parents and probe common layout ffmpeg-*/bin
    cur = start
    while True:
        cand = next(cur.glob("ffmpeg-*/bin/ffmpeg.exe"), None)
        if cand: return cand.parent
        if cur == cur.parent: break
        cur = cur.parent
    raise FileNotFoundError(f"ffmpeg.exe not found starting at {start}. "
                            f"Place an extracted ffmpeg folder (e.g., ffmpeg-8.0-essentials_build) in your project.")

ROOT  = Path.cwd()
FFBIN = find_ffbin(ROOT)


# make exes and DLLs visible to this process
os.environ["PATH"] = str(FFBIN) + ";" + os.environ.get("PATH","")
try: os.add_dll_directory(str(FFBIN))
except Exception: pass

# configure pydub
AudioSegment.converter = str(FFBIN / "ffmpeg.exe")
AudioSegment.ffprobe   = str(FFBIN / "ffprobe.exe")

# verify
subprocess.run([str(FFBIN/"ffprobe.exe"), "-version"], check=True)
# print("FFBIN =", FFBIN)



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
