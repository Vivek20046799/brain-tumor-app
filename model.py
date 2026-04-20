beta = 0.5
spike_grad = surrogate.fast_sigmoid()

class SEBlock(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, ch//reduction),
            nn.ReLU(),
            nn.Linear(ch//reduction, ch),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(x).view(x.size(0), x.size(1), 1, 1)
        return x * w

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch)
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.se = SEBlock(out_ch)

    def forward(self, x):
        out = self.conv(x)
        out += self.shortcut(x)
        return self.se(torch.relu(out))

class BigSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = ResBlock(3, 64)
        self.layer2 = ResBlock(64, 128)
        self.layer3 = ResBlock(128, 256)
        self.layer4 = ResBlock(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.lif_mid1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif_mid2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc1 = nn.Linear(512*8*8, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 4)

    def forward(self, x, T=4):
        mem_mid1 = self.lif_mid1.init_leaky()
        mem_mid2 = self.lif_mid2.init_leaky()
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        outputs = []
        x_seq = x.unsqueeze(0).repeat(T,1,1,1,1)

        for step in range(T):
            x_t = x_seq[step]

            x1 = self.pool(self.layer1(x_t))
            spk_mid1, mem_mid1 = self.lif_mid1(x1, mem_mid1)

            x2 = self.pool(self.layer2(spk_mid1))
            spk_mid2, mem_mid2 = self.lif_mid2(x2, mem_mid2)

            x3 = self.pool(self.layer3(spk_mid2))
            x4 = self.pool(self.layer4(x3))

            spk1, mem1 = self.lif1(x4, mem1)

            flat = spk1.view(spk1.size(0), -1)
            flat = self.dropout(flat)

            fc = self.fc1(flat)
            spk2, mem2 = self.lif2(fc, mem2)

            out = self.fc2(spk2)
            outputs.append(out)

        return torch.stack(outputs).mean(0)
