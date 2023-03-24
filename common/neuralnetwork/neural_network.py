# %%
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#%%
# GPU와 같은 하드웨어 가속기에서 모델을 학습한다.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

#%%
# 신경망은 데이터에 대한 연산을 수행하는 계층(layer)/ 모듈(Module)로 구성
# Pytorch의 모든 모듈은 nn.Module의 하위 클래스이며, 모듈은 입력 데이터를 받아 출력 데이터를 생성하는 forward() 메서드를 구현
class NeuralNetwork(nn.Module):
    def __init__(self):
        # init 함수에서 신경망의 계층들을 초기화
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    # nn.Module을 상속받은 모든 클래스는 forward 메소드에 입력 데이터에 대한 연산들을 구현
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
#%%
# 신경망 출력하기
model = NeuralNetwork().to(device)
print(model)

#%%
X = torch.rand(1, 28, 28, device=device)

# 모델을 사용하기 위하여 입력 데이터를 전달
# 일부 백그라운드 연산들과 함께 모델의 forward를 실행하기에 별도의 forward 호출이 필요하지 않음
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

#%%
# 모델 계층
# 사용자 정의 모델에 입력하기 위해 3개의 배치를 가지는 28*28사이즈의 이미지를 사용
input_image = torch.rand(3,28,28)
print(input_image.size())       # torch.Size([3, 28, 28])

#%%
# nn.Flatten 계층은 28*28의 2D이미지를 784 픽셀 값을 갖는 연속된 배열로 변환한다.
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())        # torch.Size([3, 784])
# %%
# nn.Linear 계층은 선형 변환을 적용하여 입력에 가중치를 곱하고 편향을 더한다.
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())           # torch.Size([3, 20])

# %%
print(f"Before ReLU: {hidden1}\n\n")    # Before ReLU: 0 미만의 수들을 존재
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")         # After ReLU: 0 미만의 수들을 0으로 만들어줌

# %%
# nn.Sequential은 모듈들을 차례대로 적용하여 출력을 생성
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)

# %%
# nn.Softmax 계층은 각 입력 샘플에 대해 클래스 확률을 계산(0~1사이의 값)
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
pred_probab
# %%
print(f"Model structure: {model}\n\n")

# nn.Module을 상속하면 모델 객체 내부의 모든 필드는 자동으로 추적
# 모델의 parameters() 및 named_parameters() 메서드를 사용하여 모델의 매개변수에 접근
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
    
# %%
