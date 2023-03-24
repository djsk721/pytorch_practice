#%%
# Torch.AutoGrad를 이용한 자동 미분
# 신경망을 학습할 때 가장 자주 사용되는 알고리즘은 역전파(Backpropagation)이다.
# 모델 가중치는 주어진 매개변수에 대한 손실 함수의 기울기(Gradient)에 비례하여 조정된다.
# Pytorch에는 torch.autograd 패키지가 있어서 모델의 학습을 위한 자동 미분을 제공한다.
# ex) 입력 x, 매개변수 w와 b를 가진 신경망을 생각해보자.
import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output

# w와 b는 최적화를 해야하는 매개변수이다.
# 따라서 기울기를 계산할 수 있어야 하기에 requires_grad=True로 설정한다.
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

# %%
# 기울기(Gradient) 계산하기
loss.backward()
print(w.grad)
print(b.grad)

#%%
# 기울기 추적 중단하기
z = torch.matmul(x, w)+b
print(z.requires_grad)

# 입력 데이터를 단순 적용인 경우와 같이 순전파 연산만 필요한 경우에는,
# 기울기 추적이나 역전파가 필요 없기에 torch.no_grad()를 사용하여 기울기 추적을 중단할 수 있다.
with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)

# 위의 torch.no_grad()와 같은 기능을 하는 코드
z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)
# %%
# autograd는 데이터의 실행된 모든 연산들의 기록을 Function 객체로 구성된
# 방향성 비순환 그래프(DAG; Directed Acyclic Graph)로 유지한다.
# 순전파 단계
# 1. 요청된 연산을 수행하여 결과 텐서를 계산하고,
# 2. DAG에 연산의 기울기 기능(Gradient Function)을 유지(maintain)한다.

# 역전파 단계
# 1. 각 .grad_fn으로부터 변화도를 계산하고,
# 2. 각 텐서의 .grad 속성에 누적한다.
# 3. 연쇄 법칙(chain rule)을 사용하여 모든 리프 텐서들까지 전파한다.

#%%
# 텐서 변화도와 야코비안 곱(Jacobian Product)
# 스칼라 손실 함수를 가지고 일부 매개변수와 관련한 변화도를 계산해야 한다.
# 그러나 출력 함수가 임의의 텐서인 경우가 있다.
# 이 경우에는 야코비안 곱(Jacobian Product)을 계산해야 한다.
# 이는 출력 텐서의 각 요소에 대해 손실 함수의 변화도를 계산하는 것이다.
inp = torch.eye(4, 5, requires_grad=True)
out = (inp+1).pow(2).t()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nSecond call\n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")