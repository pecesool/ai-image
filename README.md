# AI Image &middot; [![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python)](https://www.python.org/) [![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv)](https://opencv.org/) [![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?logo=numpy)](https://numpy.org/) [![Status](https://img.shields.io/badge/Status-Research%20Ready-success)](https://github.com/pecesool/ai-image)

> **Classical computer vision meets modern AI — a university research project exploring intelligent image processing pipelines.**  
> From traditional filtering and edge detection to neural network-driven analysis, built entirely in Python with production-grade tooling.

---

## 🎯 What This Project Delivers

**AI Image** is a university research project that bridges the gap between classical digital image processing and modern AI-driven computer vision. Rather than relying on pre-built APIs or black-box solutions, this codebase implements foundational algorithms from scratch — giving full control over every pixel transformation.

The project demonstrates how traditional techniques (Gaussian filtering, histogram equalization, edge detection) serve as the backbone for more advanced AI workflows, including segmentation, feature extraction, and intelligent image analysis.

---

## ✨ Core Features & Engineering Highlights

| Feature | What It Does | Technical Depth |
|---------|-------------|-----------------|
| **Color Space Transformations** | Convert between RGB, Grayscale, HSV, and LAB color spaces | Manual matrix operations via NumPy — no library shortcuts |
| **Spatial Filtering** | Gaussian blur, median filter, bilateral filter for noise reduction | Kernel convolution with configurable parameters and boundary handling |
| **Histogram Processing** | Equalization, normalization, and cumulative distribution analysis | Multidimensional mathematical processing for contrast enhancement |
| **Edge Detection** | Canny, Sobel, and Laplacian operators for feature extraction | Gradient magnitude computation with non-maximum suppression |
| **Image Segmentation** | Thresholding and region-based partitioning | Adaptive thresholding with Otsu's method for automatic binarization |
| **Morphological Operations** | Dilation, erosion, opening, closing for shape analysis | Structuring element design for biological/medical image preprocessing |
| **Feature Extraction** | Keypoint detection and descriptor computation | Foundation for downstream machine learning and object recognition |
| **Pipeline Architecture** | Chain operations declaratively | Modular design allowing researchers to experiment with processing sequences |

---

## 🏗️ Architecture & Design Decisions

```
ai-image/
├── main.py              # Entry point — CLI interface for running pipelines
├── core/
│   ├── filters.py       # Spatial filtering implementations
│   ├── transforms.py    # Color space and geometric transformations
│   ├── histogram.py     # Histogram-based operations
│   ├── edges.py         # Edge detection algorithms
│   ├── segmentation.py  # Image partitioning techniques
│   └── morphology.py    # Morphological image processing
├── utils/
│   ├── io.py            # Image loading, saving, format handling
│   └── visualization.py # Matplotlib wrappers for result comparison
├── pipelines/
│   └── standard.py      # Pre-configured processing chains
├── tests/
│   └── test_*.py        # Unit tests for algorithm correctness
└── requirements.txt     # Dependency manifest
```

### Why This Architecture Works

- **Algorithmic Transparency**: Every filter and transform is implemented manually with NumPy — no hidden OpenCV calls for core logic, ensuring educational and research value
- **Pipeline Pattern**: Operations are composable and chainable, enabling rapid experimentation with different preprocessing sequences
- **Test Coverage**: Unit tests validate numerical correctness against known reference implementations
- **Zero-Config Setup**: Pure Python with standard scientific stack — runs on any platform without GPU dependencies

---

## 🛠️ Technology Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| **Language** | Python 3.10+ | Rapid prototyping, extensive scientific ecosystem, readable math |
| **Numerical Computing** | NumPy | Vectorized array operations, matrix math for image kernels |
| **Image I/O** | OpenCV (cv2) | High-performance image loading, format support, display |
| **Visualization** | Matplotlib | Publication-quality plots, histograms, side-by-side comparisons |
| **Testing** | pytest | Automated validation of algorithm correctness |
| **Environment** | venv / conda | Reproducible dependency management |

---

## 🚦 Quick Start

### Prerequisites
- Python 3.10+
- pip

### Installation
```bash
# Clone repository
git clone https://github.com/pecesool/ai-image.git
cd ai-image

# Create virtual environment
python -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate
# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run
```bash
# Run the main application
python main.py

# Run tests
pytest tests/
```

> **Note**: Place input images in the `data/` directory or specify paths via command-line arguments.

---

## 📊 Example Pipeline

```python
from core.filters import gaussian_blur, median_filter
from core.edges import canny_edge_detector
from core.segmentation import otsu_threshold
from utils.io import load_image, save_image

# Load image
img = load_image("input.jpg")

# Preprocessing pipeline
blurred = gaussian_blur(img, sigma=1.5)
denoised = median_filter(blurred, kernel_size=3)

# Feature extraction
edges = canny_edge_detector(denoised, low_threshold=50, high_threshold=150)

# Segmentation
binary = otsu_threshold(denoised)

# Save results
save_image(edges, "edges_output.png")
save_image(binary, "segmentation_output.png")
```

---

## 🎯 What This Project Demonstrates

> **"This isn't a wrapper around OpenCV. It's a deep dive into how image processing actually works under the hood."**

- **Mathematical Foundations**: Understanding of linear algebra, signal processing, and statistics as applied to digital images
- **Algorithm Implementation**: Ability to translate mathematical formulas into efficient, vectorized Python/NumPy code
- **Computer Vision Literacy**: Knowledge of the complete pipeline from raw pixels to actionable features
- **Research Mindset**: Modular, testable code designed for experimentation and iteration
- **Scientific Python**: Proficiency with the NumPy/Matplotlib/OpenCV stack used across academia and industry
- **Software Engineering**: Clean module boundaries, documentation, and version control — even in research code

---

## 🎓 Academic Context

Developed as part of university coursework in **Digital Image Processing** and **Artificial Intelligence**. The project emphasizes understanding over abstraction — every algorithm is traceable to its mathematical origins.

---

**Author**: [Zhassulan Baimyshev](https://www.linkedin.com/in/zhassulan-baimyshev/)  
*University research project — Classical & AI-driven image processing in Python*
