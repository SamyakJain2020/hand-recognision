from moviepy.editor import *

# Load the video file
link = '/media/samyak/DATA/DATA/2 DO/SAMYAK/HandMotion/Dataset/Address/C0597.MP4'

video = VideoFileClip(link)

# Extract the audio and save it as a new file
audio = video.audio
audio.write_audiofile("example_audio.mp3")

# Close the video file
video.close()
