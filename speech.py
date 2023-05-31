import speech_recognition as sr
from moviepy.editor import *

# Load the video file
link = '/media/samyak/DATA/DATA/2 DO/SAMYAK/HandMotion/Dataset/Balance/C0209.MP4'

video = VideoFileClip(link)
# Extract the audio and save it as a new file
audio = video.audio
audio.write_audiofile("example_audio.wav")

# Close the video file
video.close()
# Initialize the recognizer
r = sr.Recognizer()

# Load the audio file
with sr.AudioFile("example_audio.wav") as source:
    # Record audio from the file
    audio_data = r.listen(source)
    # Convert speech to text
    try:
        text = r.recognize_google(audio_data)
    except:
        text = "Sorry, I did not get that"
    print(text)