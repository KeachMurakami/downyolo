import av
import cv2
import os
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from ultralytics import YOLO

st.title("Downy mildew detector")


# 一時フォルダ作成
TMP_DIR_PATH = "tmp"
if not os.path.exists(TMP_DIR_PATH):
    os.makedirs(TMP_DIR_PATH)

threshold = st.sidebar.slider('検出感度', 50, 100, 80)
refresh = st.sidebar.slider('フレーム数', 20, 60, 40)

# model weights for YOLOv9
weight_file = st.sidebar.file_uploader("Upload a weight (YOLOv9.pt)", type=["pt"])

class FrameSkipper(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.frame_skip = refresh  # 処理するフレームの間隔
        self.last_frame = None

    def recv(self, frame):
        self.frame_count += 1
        if self.frame_count % self.frame_skip == 1:
        # 処理対象フレームの場合
            img = frame.to_ndarray(format="bgr24")
            
            # YOLOの実行
            model = YOLO(weight_path)
            results = model.predict(img, conf = (100 - threshold)/100)
    
            # 予測結果の描画
            for point in results[0].boxes.xyxy:
                cv2.rectangle(img,
                              (int(point[0]), int(point[1])),
                              (int(point[2]), int(point[3])),
                              (0, 0, 255),
                              thickness=5)

            img = cv2.putText(img, "working", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,255,255))

            frame_out = av.VideoFrame.from_ndarray(img, format="bgr24")
            self.last_frame = frame_out
            
            return frame_out
        else:
        # 処理対象フレーム以外の場合
            if self.last_frame is not None:
                return self.last_frame
            else:
            # last_frameがNoneであれば、現在のフレームをそのまま返す
                return frame

if weight_file is not None:
    weight_path = os.path.join(TMP_DIR_PATH, weight_file.name)
    with open(weight_path, "wb") as wp:
         wp.write(weight_file.getvalue())

    webrtc_streamer(
      key="example",
      video_processor_factory=FrameSkipper,
      rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
