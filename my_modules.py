import torch.nn as nn

class ConvBlock(nn.Module):
	'''2D Conv Block 
		- out_channels = 2 * in_channels
		- halves filter dimensions '''

	def __init__(self,in_channels):
		super(ConvBlock, self).__init__()

		out_channels = in_channels * 2
		self.main = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1, bias = False),
			nn.BatchNorm2d(out_channels),
			nn.LeakyReLU(.2, inplace=True)
		)

	def forward(self,x):
		return self.main(x)

class Discriminator(nn.Module):

	def __init__(self,num_features):
		super(Discriminator, self).__init__()

		self.num_features = num_features
		self.main = nn.Sequential(

			# INPUT: bs, 3, 64, 64
			nn.Conv2d(3, self.num_features, kernel_size=4, stride=2, padding=1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# OUTPUT: bs, 64, 32, 32

			## ConvBlocks 
			# out_channels = 2 * in_channels
			# halves filter dimensions
			ConvBlock(in_channels = self.num_features),
			# OUTPUT: bs, 128, 16,16
			ConvBlock(in_channels = self.num_features * 2),
			# OUTPUT: bs, 256, 8, 8 
			ConvBlock(in_channels = self.num_features * 4),
			# OUTPUT: bs, 512,4,4

			nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
			# OUTPUT: b, 1, 1, 1
			nn.Flatten(),
			# OUTPUT: b,1

			# transform value between 0 and 1
			nn.Sigmoid()
		)

	def forward(self,img_batch):
		return self.main(img_batch)


#################################

class ConvTransposeBlock(nn.Module):

	def __init__(self,in_channels):
		super(ConvTransposeBlock, self).__init__()

		out_channels = in_channels // 2
		self.main = nn.Sequential(
			nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(True),
		)
		def forward(self,x):
			return self.main(x)

class Generator(nn.Module):

	def __init__(self, latent_vector_len, num_features):
		super(Generator, self).__init__()

		self.latent_vector_len = latent_vector_len
		self.num_features = num_features
		self.main = nn.Sequential(
			# INPUT: bs, 100, 1, 1
			# vector with langth latent_vector_len
			nn.ConvTranspose2d( latent_vector_len, num_features * 8, kernel_size = 4, stride = 1, padding = 0, bias = False),
			nn.BatchNorm2d(num_features * 8),
			nn.ReLU(True),
			# OUTPUT: bs, 800, 512, 4, 4

			## Conv2DTranspose Blocks
			ConvTransposeBlock(num_features * 8),
			# OUTPUT: bs, 256, 8, 8
			ConvTransposeBlock(num_features * 4),
			# OUTPUT: bs, 128, 16, 16
			ConvTransposeBlock(num_features * 2),
			# OUTPUT: bs, 64, 32, 32

			# Create images! (3 channels of 64x64)
			nn.ConvTranspose2d( num_features, 3, 4, 2, 1, bias=False),
			# OUTPUT: bs, 3, 64, 64
			nn.Tanh()
	)

	def forward(self,img_batch):
		return self.main(img_batch)