import torch

def CS_Score(pred, label, L):
    num = len(pred)
    dif = abs(pred - label)
    score = 0
    for i in dif:
        if i <= L:
            score += 1
    score = score / num
    return score

x = torch.rand([3])
y = torch.rand([3])
print("x", x)
print("y", y)
print("x-y", abs(x-y))
print("CS", CS_Score(x, y, 1))
