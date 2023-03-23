# %%
# 텐서는 배열이나 행력과 매우 유사한 특수한 자료구조
# 텐서를 사용하여 모델의 입력과 출력 모델의 매개변수들을 부호화
# 텐서는 GPU나 다른 하드웨어 가속기에서 실행할 수 있는 점만 빼면 NumPy의 ndarray와 매우 유사
# 텐서는 자동 미분을 지원
import torch
import numpy as np

#%%
# 텐서 초기화(torch.tensor)
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

# 텐서 초기화(numpy.ndarray)
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

#%%
x_ones = torch.ones_like(x_data) # x_data의 속성을 유지합니다.
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # x_data의 속성을 덮어씁니다.
print(f"Random Tensor: \n {x_rand} \n")

# %%
# 랜덤 텐서 할당
shape = (2,3,)
rand_tensor = torch.rand(shape) # 랜덤 텐서
ones_tensor = torch.ones(shape) # 1로 채워진 텐서
zeros_tensor = torch.zeros(shape)   # 0으로 채워진 텐서

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

#%%
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# %%
# GPU가 존재하면 텐서를 이동합니다
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)