import ffmpeg._run as ffmpeg
import torch
import cv2
from transformers import CLIPProcessor, CLIPModel
from transformers import BridgeTowerProcessor, BridgeTowerForImageAndTextRetrieval
import time
import os
from PIL import Image
import yt_dlp

COUNTER = 1  # Counter for video results
DOWNLOAD_PATH = "./website/videos"
RESULT_PATH = "./website/static/results"
IS_PROCESSING = False

def trim(input, output, start, end):
    time = end - start
    # ffmpeg.input(input, ss=start).output(output, t=time).run()
    ffmpeg.input(input, ss=start).output(output, t=time).run()
    global COUNTER
    COUNTER += 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Model Initialization (Using Large Models)
model_clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor_clip = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

processor_bt = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-gaudi")
model_bt = BridgeTowerForImageAndTextRetrieval.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-gaudi").to(device)

def clip_process_frame(frame, prompt):
    with torch.no_grad():
        inputs = processor_clip(text=[prompt, "an object"], images=frame,
                                return_tensors="pt", padding=True).to(device)
        outputs = model_clip(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).cpu().numpy()
    return probs[0][0]

def bt_process_frame(frame, prompt):
    scores = {}
    encoding = processor_bt(frame, prompt, return_tensors="pt").to(device)
    outputs = model_bt(**encoding)
    scores[prompt] = outputs.logits[0, 1].item()
    return scores

def process_video(video, prompt, n):
    capture = cv2.VideoCapture(video)
    fps = capture.get(cv2.CAP_PROP_FPS)
    print(fps)
    i = 0
    timestamp_begin = 0
    first_timestamp_found = False
    forgiving_frames = 0
    valid_frames = 0

    if n > fps:
        n = fps
    start_time = time.time()
    global IS_PROCESSING
    IS_PROCESSING = True

    while IS_PROCESSING:
        success, frame = capture.read()
        if not success:
            break

        if i % (fps // n) == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            score_clip = clip_process_frame(frame, prompt)
            print("clip:", score_clip)

            if score_clip >= 0.9 and not first_timestamp_found:
                score_bt = bt_process_frame(frame, prompt)
                print("bt:", score_bt[prompt])
                if score_bt[prompt] >= 2:
                    first_timestamp_found = True
                    timestamp_begin = i / fps
                    forgiving_frames = 0
                    valid_frames += 1
            elif score_clip >= 0.9 and first_timestamp_found:
                score_bt = bt_process_frame(frame, prompt)
                print("bt:", score_bt[prompt])
                if score_bt[prompt] < 2:
                    if forgiving_frames <= n:
                        forgiving_frames += 1
                    else:
                        if valid_frames >= n:
                            timestamp_end = (i - 1) / fps
                            first_timestamp_found = False
                            trim(video, f"{RESULT_PATH}/video{COUNTER}.mp4", timestamp_begin, timestamp_end)
                            forgiving_frames = 0
                            valid_frames = 0
                        else:
                            forgiving_frames = 0
                            valid_frames = 0
                            first_timestamp_found = False
                else:
                    valid_frames += 1
            elif score_clip < 0.9 and first_timestamp_found:
                if forgiving_frames <= n:
                    forgiving_frames += 1
                else:
                    if valid_frames >= n:
                        timestamp_end = (i - 1) / fps
                        first_timestamp_found = False
                        trim(video, f"{RESULT_PATH}/video{COUNTER}.mp4", timestamp_begin, timestamp_end)
                        forgiving_frames = 0
                        valid_frames = 0
                    else:
                        forgiving_frames = 0
                        valid_frames = 0
                        first_timestamp_found = False

        i += 1

    if first_timestamp_found:
        timestamp_end = (i - 1) / fps
        trim(video, f"{RESULT_PATH}/video{COUNTER}.mp4", timestamp_begin, timestamp_end)

    capture.release()
    print("Processing time:", time.time() - start_time)

def downloader(links):
    video_names = []
    
    os.makedirs(DOWNLOAD_PATH, exist_ok=True)

    for link in links:
        try:
            ydl_opts = {
                'outtmpl': f"{DOWNLOAD_PATH}/%(id)s.%(ext)s",
                'format': 'mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best',
                'merge_output_format': 'mp4',
                'noplaylist': True,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(link, download=True)
                video_filename = f"{info['id']}.mp4"

                if os.path.exists(os.path.join(DOWNLOAD_PATH, video_filename)):
                    print("Downloaded:", video_filename)
                    video_names.append(video_filename)
                else:
                    print("Failed to download:", link)

        except Exception as e:
            print(f"Error downloading {link}: {e}")

    return video_names

def process_links(links, prompt, fps):
    for video in os.listdir(RESULT_PATH):
        os.remove(os.path.join(RESULT_PATH, video))

    names = downloader(links)
    print("Downloaded videos:", names)

    for name in names:
        process_video(f"{DOWNLOAD_PATH}/{name}", prompt, fps)

def process_custom_videos(names, prompt, fps):
    for video in os.listdir(RESULT_PATH):
        os.remove(os.path.join(RESULT_PATH, video))

    for name in names:
        process_video(f"{DOWNLOAD_PATH}/{name}", prompt, fps)

def stop_processing():
    global IS_PROCESSING
    IS_PROCESSING = False

#"https://youtu.be/DRMzxsjWtPQ?si=ssmBJCcf0trMUq_h"
#"https://youtu.be/AJWpvoXP5d4?si=CdJXMBZKmrG6A3Yc"