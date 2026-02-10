# AprilTag Interface

一个基于Python的AprilTag检测和定位系统，支持图片和摄像头实时检测，提供精确的位姿估计。

## 功能特性

- 🎯 **高精度检测**: 基于pupil_apriltags库的AprilTag检测
- 📷 **多输入模式**: 支持静态图片和实时摄像头检测
- 📐 **位姿估计**: 提供6DOF位姿（位置+姿态）
- 🎨 **可视化**: 实时显示检测结果和详细信息
- ⚙️ **相机标定**: 支持自定义相机内参和畸变校正
- 🔧 **VSCode调试**: 完整的调试配置支持

## 项目结构

```
AprilTagInterface/
├── src/
│   ├── main.py              # 主程序入口
│   ├── detector.py          # AprilTag检测器核心
│   └── __init__.py
├── calibration/
│   ├── generate_calib.py    # 相机标定工具
│   └── realsense_calib.npz  # 标定文件示例
├── images/                  # 测试图片目录
├── .vscode/                 # VSCode调试配置
│   ├── launch.json
│   └── settings.json
├── requirements.txt         # Python依赖
└── README.md
```

## 环境要求

- Python 3.8+
- OpenCV 4.x
- NumPy
- pupil_apriltags

## 安装步骤

1. **克隆仓库**
```bash
git clone <repository-url>
cd AprilTagInterface
```

2. **创建虚拟环境**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

## 使用方法

### 图片模式

```bash
python src/main.py --tag-size 0.11 --image images/test.jpg --calib calibration/realsense_calib.npz
```

### 摄像头模式

```bash
python src/main.py --tag-size 0.11 --camera 0 --calib calibration/realsense_calib.npz
```

### 参数说明

| 参数 | 说明 | 必需 | 示例 |
|------|------|------|------|
| `--tag-size`, `-s` | Tag实际边长(米) | ✅ | `0.11` (11cm) |
| `--image`, `-i` | 图片路径 | ❌ | `images/test.jpg` |
| `--calib` | 标定文件路径 | ❌ | `calibration/camera_calib.npz` |
| `--camera`, `-c` | 摄像头ID | ❌ | `0` |
| `--output`, `-o` | 结果保存路径 | ❌ | `output.jpg` |

## 相机标定

使用内置的标定工具：

```bash
python calibration/generate_calib.py
```

按照提示采集棋盘格图像，程序会自动生成标定文件 `camera_calib.npz`。

## VSCode调试

项目已配置完整的VSCode调试环境：

1. **激活虚拟环境**:
   ```bash
   source venv/bin/activate
   ```

2. **开始调试**:
   - 按 `F5` 或点击调试按钮
   - 选择调试配置
   - 支持断点调试、变量监视等

3. **调试配置选项**:
   - 🔧 **"AprilTag: 选择图片"** - 动态输入图片路径（推荐）
   - 📸 **"AprilTag: 图片1 (155134)"** - 调试第一张测试图片
   - 📸 **"AprilTag: 图片2 (155317)"** - 调试第二张测试图片
   - 🎥 **"AprilTag: 摄像头模式"** - 实时摄像头调试

4. **快速选择图片**:
   - 选择 "AprilTag: 选择图片" 配置
   - 调试启动时会弹出输入框
   - 输入图片路径，如：`images/my_test.jpg`
   - 或使用快捷键 `Ctrl+Shift+P` → "Debug: Select and Start Debugging"

5. **调试技巧**:
   - 在 `detector.py` 和 `main.py` 中设置断点
   - 使用变量监视器查看检测结果
   - 调用堆栈跟踪函数执行流程
   - 集成终端显示实时输出

## 输出信息

检测完成后会输出每个Tag的详细信息：

```
检测到 2 个 AprilTag:

--- Tag 1 ---
ID: 0
距离: 1.2345 m
平移向量 (相机坐标系): [0.1234, 0.5678, 1.2345]
欧拉角 (度): 滚转=2.34, 俯仰=-1.23, 偏航=45.67
图像中心: (320.5, 240.8)
```

## 可视化功能

- 🎯 实时绘制Tag边界框
- 📍 显示Tag中心点
- 📊 显示距离和ID信息
- 🎨 不同颜色区分不同Tag
- 📐 可选坐标系可视化

## 快捷键（摄像头模式）

- `q` - 退出程序
- `s` - 保存当前截图

## 故障排除

### 1. 检测不到Tag
- 检查光照条件
- 确认Tag尺寸参数正确
- 验证相机标定文件

### 2. 相机打不开
- 检查摄像头ID是否正确
- 确认摄像头权限
- 尝试不同的后端（V4L2/DSHOW）

### 3. 标定文件错误
- 重新运行相机标定
- 检查文件路径和格式

## 技术细节

### 坐标系说明
- **相机坐标系**: X轴向右，Y轴向下，Z轴向前
- **欧拉角顺序**: Roll (X) → Pitch (Y) → Yaw (Z)
- **单位**: 距离为米，角度为度

### 检测算法
- 使用 `pupil_apriltags` 库进行检测
- 支持多种Tag家族（tag36h11, tag25h9等）
- 亚像素精度角点检测

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 更新日志

### v1.0.0
- 初始版本发布
- 支持图片和摄像头检测
- 完整的VSCode调试配置
- 相机标定工具