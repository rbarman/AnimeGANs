from datetime import datetime
from my_modules import Discriminator, Generator
from device_utils import get_default_device, DeviceDataLoader, to_device
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from tqdm.notebook import tqdm
import os

class GANTrainer():
	def __init__(self
		, train_dl
		, generator
		, discriminator
		, optimizerG
		, optimizerD
		, prev_session_dir = None
		, base_dir = '/content/drive/My Drive/AnimeGANs/'
		):

		self.device = get_default_device()
		self.train_dl = train_dl
		self.batch_size = train_dl.dl.batch_size
		self.base_dir = base_dir
		self.total_epoch_count = 0

		self.generator = generator
		self.discriminator = discriminator
		self.optimizerG = optimizerG
		self.optimizerD = optimizerD

		# brand new training session, so create new folders to store run data
		if prev_session_dir is None:

			# create a folder for this session
			self.session_dir = f"{base_dir}{datetime.now().strftime('%m%d%Y_%H%M%S')}/"
			os.makedirs(self.session_dir, exist_ok=True)

			# gen_dir stores images that gets generated at end of each epoch
			self.gen_dir = f'{self.session_dir}generated/'
			os.mkdir(self.gen_dir)

			# checkpoint_dir stores weights called on each checkpoint
			self.checkpoint_dir = f'{self.session_dir}checkpoints/'
			os.mkdir(self.checkpoint_dir)

		# reference dirs to previous training session
		else:
			self.session_dir = prev_session_dir
			self.gen_dir = f'{self.session_dir}generated/'
			self.checkpoint_dir = f'{self.session_dir}checkpoints/'

		print(f'Saving training session data to {self.session_dir}')
		print(f'\t{self.gen_dir} stores generated images')
		print(f'\t{self.checkpoint_dir} stores checkpoint weights')


	def train_generator(self):
		self.generator.zero_grad()

		# create fake images from the generator
		noise = torch.randn(self.batch_size, self.generator.latent_vector_len, 1, 1, device=self.device)
		fake_images = self.generator(noise)

		# pass fake images through discriminator
		predictions_fake = self.discriminator(fake_images)
		labels = torch.ones(self.batch_size,1,device=self.device)
		loss_fake = F.binary_cross_entropy(predictions_fake, labels)

		# backwards pass + update generator weights
		loss_fake.backward()
		self.optimizerG.step()

		return loss_fake.item()

	def save_generated_samples(self, name):
		''' Save generated images from Generator to self.gen_dir
		    - This could be replaced with tensor board?'''

		noise = torch.randn(self.batch_size, self.generator.latent_vector_len, 1, 1, device=self.device)
		fake_images = self.generator(noise)
		fake_path = f'{self.gen_dir}{name}.png'

		save_image(make_grid(fake_images[:64], padding=2, normalize=True,nrow=8),fake_path)
		print(f'Saved generated images to {fake_path}')

	def train_discriminator(self,images):

		# https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
		self.discriminator.zero_grad()

		# pass through real images
		predictions_real = self.discriminator(images)
		labels = torch.ones(images.shape[0],1,device=self.device)
		loss_real = F.binary_cross_entropy(predictions_real, labels)

		# create fake images from the generator
		noise = torch.randn(images.shape[0], self.generator.latent_vector_len, 1, 1, device=self.device)
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
			self.total_epoch_count = self.total_epoch_count + 1
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

			self.save_generated_samples(self.total_epoch_count)

			# Log losses & scores (last batch)
			print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
				epoch+1, num_epochs, loss_g, loss_d, real_score, fake_score))
	  
		return losses_g, losses_d, real_scores, fake_scores
  
	def save_checkpoint(self):
		''' Save generator and discrimniator models and weights to self.checkpoint_dir'''

		save_path = f'{self.checkpoint_dir}epoch{self.total_epoch_count}.pth'
		torch.save({
			''
			'epoch': self.total_epoch_count,
			'discriminator_state_dict': self.discriminator.state_dict(),
			'optimizerD_state_dict': self.optimizerD.state_dict(),
			'generator_state_dict': self.generator.state_dict(),
			'optimizerG_state_dict': self.optimizerG.state_dict()
		}
		, save_path)
		print(f'Saved model and optimzer weights to {save_path}')
