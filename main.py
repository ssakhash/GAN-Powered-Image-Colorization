# Import required libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Hyperparameters
batch_size = 64
z_dim = 100
num_epochs = 50
learning_rate = 0.0002

# Generator model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 3 * 32 * 32),  # Output size for CIFAR-10 images
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.model(x)

# Discriminator model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3 * 32 * 32, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Initialize models and optimizers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)
discriminator = Discriminator().to(device)
optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate)

# Loss function
criterion = nn.BCELoss()

# Load CelebA dataset
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = torchvision.datasets.CelebA(root='./data', split='train', transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Learning rate scheduling
scheduler_g = optim.lr_scheduler.StepLR(optimizer_g, step_size=20, gamma=0.5)
scheduler_d = optim.lr_scheduler.StepLR(optimizer_d, step_size=20, gamma=0.5)

# Training loop
for epoch in range(num_epochs):
    for batch_idx, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)
        batch_size = real_images.shape[0]

        # Train Discriminator
        optimizer_d.zero_grad()
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        # Real images
        real_outputs = discriminator(real_images.view(batch_size, -1))
        d_loss_real = criterion(real_outputs, real_labels)
        d_loss_real.backward()

        # Fake images
        z = torch.randn(batch_size, z_dim).to(device)
        fake_images = generator(z)
        fake_outputs = discriminator(fake_images.detach())
        d_loss_fake = criterion(fake_outputs, fake_labels)
        d_loss_fake.backward()

        d_loss = d_loss_real + d_loss_fake
        optimizer_d.step()

        # Train Generator
        optimizer_g.zero_grad()
        fake_outputs = discriminator(fake_images)
        g_loss = criterion(fake_outputs, real_labels)
        g_loss.backward()
        optimizer_g.step()

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(dataloader)}] "
                  f"D Loss: {d_loss.item():.4f} G Loss: {g_loss.item():.4f}")
            
    # Visualize generated images and save them every 5 epochs
    if (epoch + 1) % 5 == 0:
        with torch.no_grad():
            z_sample = torch.randn(1, z_dim).to(device)
            generated_image = generator(z_sample).view(3, 64, 64).cpu().numpy()
            generated_image = (generated_image + 1) / 2.0  # Denormalize

        plt.imshow(generated_image.transpose(1, 2, 0))
        plt.axis('off')
        plt.savefig(f'generated_image_epoch_{epoch+1}.png')
        plt.close()

        print(f"Generated image saved for epoch {epoch+1}")

# Save models
torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')

# Load a pre-trained generator
generator = Generator().to(device)  # Move the generator to the appropriate device
generator.load_state_dict(torch.load('generator.pth', map_location=device))
generator.eval()

# Generate and display a sample image
with torch.no_grad():
    z_sample = torch.randn(1, z_dim).to(device)
    generated_image = generator(z_sample).view(3, 32, 32).cpu().numpy()
    generated_image = (generated_image + 1) / 2.0  # Denormalize

# Load a sample real image
sample_real_image, _ = next(iter(dataloader))
sample_real_image = sample_real_image[0].numpy().transpose(1, 2, 0)
sample_real_image = (sample_real_image + 1) / 2.0  # Denormalize

# Display input and generated images
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title('Input (Grayscale)')
plt.imshow(sample_real_image[:, :, 0], cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Generated (Color)')
plt.imshow(generated_image.transpose(1, 2, 0))
plt.axis('off')

plt.tight_layout()
plt.show()
