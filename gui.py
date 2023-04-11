import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from pred import predict_video

# Function to handle button click event
def play_video():
    # Get the selected file path
    file_path = filedialog.askopenfilename()
    
    # Initialize the window
    res = predict_video(file_path, 15)
    
    # Create a new window for displaying video
    video_window = tk.Toplevel()
    video_window.title("Video Player")
    video_window.configure(bg="#212121")
    
    # Open the video file using OpenCV
    cap = cv2.VideoCapture(file_path)
    
    # Get the size of the video frame
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create a canvas for displaying the video
    canvas_video = tk.Canvas(video_window, width=width, height=height, bg="#212121")
    canvas_video.pack()
    
    # Create a label for displaying the prediction result
    label_result = tk.Label(video_window, text="", fg="white", bg="#212121", font=("Helvetica", 18))
    label_result.pack(pady=10)
    
    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        
        if not ret:
            # End of video
            break
        
        # Add the prediction result to the frame
        #cv2.putText(frame, f'{res}', (50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0), thickness=2)
        
        # Display the frame on the canvas
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        photo = tk.PhotoImage(master=canvas_video, data=cv2.imencode('.png', img)[1].tobytes())
        canvas_video.create_image(0, 0, anchor=tk.NW, image=photo)
        
        # Update the prediction result label
        if res == 'Violence':
            label_result.config(text=f"{res}", fg='#FF0000')
        else:
            label_result.config(text=f"{res}", fg='#00FF00')
        # Wait for a short time
        video_window.update()
        video_window.after(50)
    
    # Release the video and close the window
    cap.release()
    video_window.destroy()

# Create the main window
root = tk.Tk()
root.title("Select File")
root.configure(bg="#212121")

# Create a button for selecting a video file
select_button = tk.Button(root, text="Select Video", command=play_video, fg="white", bg="#757575", font=("Helvetica", 14), padx=20, pady=10)
select_button.pack(pady=20)

# Run the main event loop
root.mainloop()
