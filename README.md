# 📌 포트폴리오 요약

- **기간**: (시작일) ~ (최종 수정일)
- **인원**:
- **담당 역할**:

## 1. 프로젝트 개요 (Overview)

스마트폰의 IP Webcam 앱으로 스트리밍되는 영상을 수신해 등록된 인물의 얼굴을 실시간으로 인식하는 프로젝트입니다.

## 2. 기술 스택 (Tech Stack)

- Python, OpenCV
- face_recognition (dlib 기반)
- NumPy, PyAutoGUI

## 3. 핵심 기능 및 담당 구현 사항 (Key Features & Contributions)

- IP Webcam 스트림(`http://{ip}:8080/video`)을 OpenCV `VideoCapture`로 수신
- `knowns/` 폴더의 얼굴 이미지로 등록 인물 얼굴 인코딩 생성
- 매 프레임 얼굴 검출·인코딩 후 `face_distance`로 최근접 인물 매칭, 바운딩 박스와 이름 오버레이 출력

## 4. 트러블슈팅 및 문제 해결 (Troubleshooting)

- 인식된 얼굴이 2명 이상일 때 NumPy 배열(`distances`)을 `if distances:`로 boolean 평가해 `ValueError`로 크래시하던 버그를 `len(distances) > 0`으로 수정했습니다.

## 5. 실행 방법 (How to Run)

아래 원본 README의 설치/실행 순서를 참고하세요.

## 6. 성과 및 회고 (Results & Retrospective)

- (작성 예정)

---

# 원본 README

IP Webcam 어플리케이션을 활용한 얼굴인식 모델

파이썬 구버전 필요 7,8,9 
패키지 설치 
pip install opencv-python opencv-contrib-python dlib face_recognition pyautogui

1. knowns 폴서 생성
2. webcam app 실행
   
  ![KakaoTalk_20240208_141853486_02](https://github.com/SqurareEgg/IPWebCam_FaceRecognation/assets/148935595/a2f3cf27-75b0-4011-afdd-307c4c36447b)
  
4. webcam 서버 구동
5. py 실행
6. webcam app의 아이피 입력
   
   ![KakaoTalk_20240208_141853486](https://github.com/SqurareEgg/IPWebCam_FaceRecognation/assets/148935595/ab7ff96d-e907-45bd-876d-3612a6cc6df8)
   ![image](https://github.com/SqurareEgg/IPWebCam_FaceRecognation/assets/148935595/fe0aa825-a0b9-48fc-9cd1-7fe22c0f4790)
