import cv2
import os

def extractFramesFromVid(save_interval = 0.1):
    while True:
        input_path = input("Paste Path Of Video To Splice\n")
        if input_path[0] == '"':
            input_path = input_path[1:-1]

        if os.path.exists(input_path):
            break
        else:
            print("Invalid Input: Path Does Not Exist!\n")
            continue
    
    while True:
        output_path = input("Paste Path to output frame images\n")
        if output_path[0] == '"':
            output_path = output_path[1:-1]
        os.makedirs(output_path, exist_ok=True)
        break
    
    filename = os.path.basename(output_path.strip("/\\"))
    capture = cv2.VideoCapture(input_path)

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * save_interval)
    saved_count = 0
    frame_count = 0

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        
        frame_count += 1
        
        if frame_count % frame_interval == 0:
            saved_count += 1
            cv2.imwrite(output_path + "/" + f'{filename}_%d.jpg' % saved_count, frame)
            print(f'Saved Frame {saved_count}')
            continue
    
    capture.release()
    cv2.destroyAllWindows()

extractFramesFromVid()