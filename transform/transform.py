#%%
# 변형(transform) 을 해서 데이터를 조작하고 학습에 적합하게 만듭니다.
# 모든 TorchVision 데이터셋들은 변형 로직을 갖는, 
# 호출 가능한 객체(callable)를 받는 매개변수 두개 ( 특징(feature)을 변경하기 위한 transform 과 정답(label)을 변경하기 위한 target_transform )를 갖습니다 torchvision.transforms 모듈은 주로 사용하는 몇가지 변형(transform)을 제공합니다.
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

# %% 
# ToTensor 는 PIL Image나 NumPy ndarray 를 FloatTensor 로 변환하고, 
# 이미지의 픽셀의 크기(intensity) 값을 [0., 1.] 범위로 비례하여 조정(scale)합니다.
target_transform = Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
# %%
target_transform
# %%
