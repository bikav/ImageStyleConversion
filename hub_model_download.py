"""下载hub模型"""

import kagglehub

path = kagglehub.model_download("google/arbitrary-image-stylization-v1/tensorFlow1/256")

print("Path to model files:", path)
