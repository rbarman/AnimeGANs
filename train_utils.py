from datetime import datetime
from my_modules import Discriminator, Generator
from device_utils import get_default_device, DeviceDataLoader, to_device
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from tqdm.notebook import tqdm
import os

class GANTrainer():
  def __init__(self, train_dl
               , num_generator_features = 64
               , num_discriminator_feataures = 64
               ,latent_size = 100
               , lr=.0002
               , base_dir = '/content/drive/My Drive/AnimeGANs/'
               ):
    
    self.device = get_default_device()
    self.train_dl = train_dl
    self.batch_size = train_dl.dl.batch_size
    self.latent_size = latent_size
    self.num_generator_features = num_generator_features
    self.num_discriminator_feataures = num_discriminator_feataures
    self.base_dir = base_dir

    # create a folder for this session
    self.session_folder = f"{base_dir}{datetime.now().strftime('%m%d%Y_%H%M%S')}/"
    print(f'Saving training session data to {self.session_folder}')
    os.makedirs(SESSION_FOLDER, exist_ok=True)
    # gen_dir stores images that gets generated at end of each epoch
    self.gen_dir = f'{self.session_folder}generated/'
    os.mkdir(self.gen_dir)

    # set up Generator and Discriminators and put on device
    self.discriminator = Discriminator(num_features = self.num_generator_features)
    self.generator = Generator(latent_vector_len = self.latent_size,num_features = self.num_discriminator_feataures)
    self.discriminator = to_device(self.discriminator, self.device)
    self.generator = to_device(self.generator, self.device)

    # set up optimizers
    self.optimizerD = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    self.optimizerG = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))

  def train_generator(self):
    self.generator.zero_grad()

    # create fake images from the generator
    noise = torch.randn(self.batch_size, self.latent_size, 1, 1, device=self.device)
    fake_images = self.generator(noise)

    # pass fake images through discriminator
    predictions_fake = self.discriminator(fake_images)
    labels = torch.ones(self.batch_size,1,device=self.device)
    loss_fake = F.binary_cross_entropy(predictions_fake, labels)

    # backwards pass + update generator weights
    loss_fake.backward()
    self.optimizerG.step()

    return loss_fake.item()

  def save_generated_samples(self,iter):
    ''' Save generated images from Generator to disk
        - iter = training iteration number
        - This could be replaced with tensor board?
    '''

    noise = torch.randn(self.batch_size, self.latent_size, 1, 1, device=self.device)
    fake_images = self.generator(noise)
    fake_name = f'{iter}.png'

    save_image(make_grid(fake_images[:64], padding=2, normalize=True,nrow=8),
              f'{self.gen_dir}{fake_name}')
    
    print(f'Saving to {gen_dir}{fake_name}')

  def train_discriminator(self,images):

    # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
    self.discriminator.zero_grad()

    # pass through real images
    predictions_real = self.discriminator(images)
    labels = torch.ones(images.shape[0],1,device=self.device)
    loss_real = F.binary_cross_entropy(predictions_real, labels)

    # create fake images from the generator
    noise = torch.randn(images.shape[0], self.latent_size, 1, 1, device=self.device)
    fake_images = self.generator(noise)
    
    # pass through fake images
    predictions_fake = self.discriminator(fake_images)
    labels = torch.zeros(images.shape[0],1,device=self.device)
    loss_fake = F.binary_cross_entropy(predictions_fake, labels)

    # backwards pass and update discriminator weights
      # total_loss = loss_fake + loss_real
      # total_loss.backward() should be the same thing (?)
    loss_fake.backward()
    loss_real.backward()
    total_loss = loss_fake + loss_real
    
    self.optimizerD.step()

    # return total loss, 'score' of real predictions, and 'score' of fake predictions
    return total_loss.item(), torch.mean(predictions_real).mean(), torch.mean(predictions_fake).mean()

  def train(self,num_epochs, iter_start=1):

    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []
    
    for epoch in range(num_epochs):
      for images, _ in tqdm(self.train_dl):

        # Train discriminator
        loss_d, real_score, fake_score = self.train_discriminator(images)
        # Train generator
        loss_g = self.train_generator()

      # Record losses & scores
      losses_g.append(loss_g)
      losses_d.append(loss_d)
      real_scores.append(real_score)
      fake_scores.append(fake_score)
      
      self.save_generated_samples(epoch+iter_start)

      # Log losses & scores (last batch)
      print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
          epoch+1, num_epochs, loss_g, loss_d, real_score, fake_score))
      
    return losses_g, losses_d, real_scores, fake_scores
