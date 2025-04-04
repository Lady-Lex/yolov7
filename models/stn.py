import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleSTN(nn.Module):
    def __init__(self):
        super(SimpleSTN, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7, padding=3),  # 保证尺寸不变
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5, padding=2),  # 保证尺寸不变
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))  # 输出固定大小

        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 4, 32),
            nn.ReLU(True),
            nn.Linear(32, 6)
        )

        # 初始化仿射矩阵为单位矩阵
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)          # 输出 shape: [B, 10, H', W']
        xs = self.global_pool(xs)          # 强制变为 [B, 10, 4, 4]
        xs = xs.view(xs.size(0), -1)       # 变为 [B, 160]
        theta = self.fc_loc(xs).view(-1, 2, 3)  # [B, 2, 3]

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x
