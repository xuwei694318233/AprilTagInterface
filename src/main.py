#!/usr/bin/env python3
"""
AprilTag 定位 - 支持图片和摄像头
"""

import argparse
import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from detector import AprilTagDetector


def load_calibration(calib_path: str):
    """加载相机标定文件"""
    data = np.load(calib_path)
    return data['mtx'], data.get('dist', np.zeros(4))


def main():
    parser = argparse.ArgumentParser(description='AprilTag 定位')
    parser.add_argument('--tag-size', '-s', type=float, required=True, 
                       help='Tag实际边长(米)，如0.165表示16.5cm')
    parser.add_argument('--image', '-i', type=str,
                       help='图片路径（不填则使用摄像头）')
    parser.add_argument('--calib', type=str, default='calibration/camera_calib.npz',
                       help='标定文件路径')
    parser.add_argument('--camera', '-c', type=int, default=0,
                       help='摄像头ID（图片模式不需要）')
    parser.add_argument('--output', '-o', type=str,
                       help='保存结果图片路径')

    args = parser.parse_args()

    # 加载相机内参
    if Path(args.calib).exists():
        camera_matrix, dist_coeffs = load_calibration(args.calib)
        print(f"已加载标定文件: {args.calib}")
    else:
        print(f"警告: 未找到标定文件 {args.calib}，使用默认内参")
        # 根据常见分辨率预设
        camera_matrix = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.zeros(4)

    print(f"内参矩阵:\n{camera_matrix}")

    # 初始化检测器
    detector = AprilTagDetector(args.tag_size, camera_matrix, dist_coeffs)

    # 图片模式
    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            print(f"错误: 无法读取图片 {args.image}")
            return

        print(f"图片尺寸: {image.shape[1]}x{image.shape[0]}")

        # 检测
        results = detector.detect(image)

        print(f"\n检测到 {len(results)} 个 AprilTag:")
        for i, r in enumerate(results):
            print(f"\n--- Tag {i+1} ---")
            print(f"ID: {r['id']}")
            print(f"距离: {r['distance']:.4f} m")
            print(f"平移向量 (相机坐标系): [{r['tvec'][0]:.4f}, {r['tvec'][1]:.4f}, {r['tvec'][2]:.4f}]")
            print(f"欧拉角 (度): 滚转={r['euler'][0]:.2f}, 俯仰={r['euler'][1]:.2f}, 偏航={r['euler'][2]:.2f}")
            print(f"图像中心: ({r['center'][0]:.1f}, {r['center'][1]:.1f})")

        # 可视化
        vis = detector.draw_detections(image, results)

        # 添加文字信息
        info_text = f"Tags: {len(results)} | Tag size: {args.tag_size*100:.1f}cm"
        cv2.putText(vis, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 保存或显示
        if args.output:
            cv2.imwrite(args.output, vis)
            print(f"\n结果已保存: {args.output}")
        else:
            # 窗口显示
            window_name = "AprilTag Detection - Press any key to close"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1280, 720)
            cv2.imshow(window_name, vis)
            print("\n按任意键关闭窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return

    # 摄像头模式（原有代码）
    print("启动摄像头模式...")
    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    print("按 'q' 退出，'s' 保存截图")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = detector.detect(frame)
        vis = detector.draw_detections(frame, results)
        
        cv2.imshow("AprilTag", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("capture.jpg", vis)
            print("已保存 capture.jpg")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
