import os
import json
import math
import time
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
from scipy.io import wavfile

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
from text import text_to_sequence, cleaned_text_to_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

hps = utils.get_hparams_from_file("./configs/ljs_mb_istft_vits.json")
os.makedirs("test_audio", exist_ok=True)

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).to(device)
_ = net_g.eval()

_ = utils.load_checkpoint("/nfs1/yi.liu/tts/mb-istft-vits/pretrained_MB-iSTFT-VITS_ddp.pth.pth", net_g, None)

# stn_tst = get_text("This is a sample audio", hps)
# with torch.no_grad():
#     x_tst = stn_tst.cuda().unsqueeze(0)
#     x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
#     audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()

# warm-up
print("Warming up...")
audiopaths_and_text = utils.load_filepaths_and_text('filelists/warmup.txt')
with torch.no_grad():
    for audiopath, text in audiopaths_and_text:
        text_norm = cleaned_text_to_sequence(text)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_padded = torch.LongTensor(text_norm)
        x_tst = text_padded.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([text_padded.size(0)]).to(device)
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()

# do actual work
audiopaths_and_text = utils.load_filepaths_and_text('filelists/ljs_audio_text_test_filelist.txt.cleaned')
use_time = 0
wav_time = 0
print("Use MB-ISTFT-VITS (DDP)")
with torch.no_grad():
    for audiopath, text in audiopaths_and_text:
        print(audiopath)
        text_norm = cleaned_text_to_sequence(text)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_padded = torch.LongTensor(text_norm)
        x_tst = text_padded.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([text_padded.size(0)]).to(device)

        start_time = time.time()
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
        use_time += time.time() - start_time

        audio *= 32767 / max(0.01, np.max(np.abs(audio))) * 0.6
        audio = np.clip(audio, -32767.0, 32767.0)
        wav_name = audiopath.split("/")[-1]
        wavfile.write("test_audio/" + wav_name,
                      hps.data.sampling_rate, audio.astype(np.int16))
        wav_time += audio.shape[0] / hps.data.sampling_rate
print(f"Total synthesized time: {use_time}")
print(f"Total wav time: {wav_time}")
print(f"RTF: {use_time/wav_time}")
