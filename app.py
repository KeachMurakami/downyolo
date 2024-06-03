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
weight_file = st.sidebar.file_uploader("Upload a weight (YOLOv9.pt)", type=["pt"])

tab1, tab2 = st.tabs(["画像 📷", "動画 🎥"])

with tab1:
    image_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if image_file is not None:
        # アップロードした画像の保存
        file_path = os.path.join(TMP_DIR_PATH, image_file.name)
        with open(file_path, "wb") as fp:
            fp.write(image_file.getvalue())
    
        if weight_file is not None:
             # アップロードした画像の保存
            weight_path = os.path.join(TMP_DIR_PATH, weight_file.name)
            with open(weight_path, "wb") as wp:
                 wp.write(weight_file.getvalue())
    
            # YOLOの実行
            model = YOLO(weight_path)
            results = model.predict(file_path, save=True, conf = (100-threshold)/100)
    
            # 予測結果の描画
            img = cv2.imread(file_path)
            for point in results[0].boxes.xyxy:
                cv2.rectangle(img,
                              (int(point[0]), int(point[1])),
                              (int(point[2]), int(point[3])),
                              (0, 0, 255),
                              thickness=5)
    
        # 解析画像の保存
            analysis_img_path = os.path.join(TMP_DIR_PATH,
                                             f"analysis_{image_file.name}")
            cv2.imwrite(analysis_img_path, img)
    
        # 画像の表示
            st.image(analysis_img_path,
                     use_column_width=True)
       

with tab2:
    # refresh = st.slider('更新速度 (fps)', 1, 3, 1)
    refresh = st.radio(
        "更新スピード",
        ["normal", ":rainbow[fast (experimental)]"],
        captions = ["毎秒更新", "フレームごとに更新 (通信量が大きいのでオンライン使用は非推奨)"],
        index=0
    )
    
    # model weights for YOLOv9
    
    class FrameSkipper(VideoProcessorBase):
        def __init__(self):
            self.frame_count = 0
            self.last_frame = None
            
            if refresh == "normal":
                self.frame_skip = 90  # 処理するフレームの間隔
            else:
                self.frame_skip = 10  # 処理するフレームの間隔
    
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
