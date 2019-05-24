import numpy as np
import onnx

model_path = './model.onnx'

import onnxruntime as ort
sess = ort.InferenceSession(model_path)

obs = np.array([-0.02769702, 0.02838421, 0.0245941, -0.00242773]).reshape([1, 4]).astype(np.float32)

res = sess.run([], {'default_policy/obs:0': obs})

print(res)
