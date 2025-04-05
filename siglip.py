"""
siglip.py

This script demonstrates how to preprocess an image, create patch embeddings,
and build a Siglip-style vision transformer using PyTorch and the Hugging Face
Transformers library. The script includes detailed examples of patch embedding
visualization, attention head implementation, MLP blocks, and assembling the
complete transformer model. It also shows how to load pretrained weights from
a Hugging Face model and verify that the outputs match.
"""

from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass
from torchvision import transforms
import math
import matplotlib.pyplot as plt

# ---------------------------
# Image Loading and Preprocessing
# ---------------------------

# Load an image using PIL.
img = Image.open("cat_image.jpg")
img

# Load processor and model from the Hugging Face Transformers library.
from transformers import AutoProcessor, SiglipVisionModel, SiglipVisionConfig

processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
vision_model = SiglipVisionModel.from_pretrained(
    "google/siglip-base-patch16-224",
    config=SiglipVisionConfig(vision_use_head=False)
)

print(vision_model)

def preprocess_image(image, image_size=224):
    """
    Preprocess an image by resizing, converting to tensor, and normalizing.

    Args:
        image (PIL.Image.Image): Input image.
        image_size (int): Desired image size (default: 224).

    Returns:
        torch.Tensor: Preprocessed image tensor with shape (1, 3, image_size, image_size).
    """
    preprocess = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image_tensor = preprocess(image)
    # Add batch dimension: (3, H, W) -> (1, 3, H, W)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor

# Preprocess the loaded image.
image_tensor = preprocess_image(img)

# ---------------------------
# Creating Patch Embeddings
# ---------------------------

# Define parameters for patch embedding.
embed_dim = 768
patch_size = 16
image_size = 224
num_patches = (image_size // patch_size) ** 2

# Create patch embeddings using a convolution layer.
with torch.no_grad():
    patch_embedding = nn.Conv2d(
        in_channels=3,
        out_channels=embed_dim,
        kernel_size=patch_size,
        stride=patch_size
    )
    patches = patch_embedding(image_tensor)

print(patches.shape, num_patches)

# Create a position embedding to add positional information.
position_embedding = nn.Embedding(num_patches, embed_dim)
position_ids = torch.arange(num_patches).expand((1, -1))
print(position_ids.shape)

# Flatten and transpose patch embeddings to get shape (batch_size, num_patches, embed_dim).
embeddings = patches.flatten(start_dim=2, end_dim=-1)  # from (1, 768, 196)
embeddings = embeddings.transpose(1, 2)                 # to (1, 196, 768)

# Add positional embeddings.
embeddings = embeddings + position_embedding(position_ids)
print(embeddings.shape)

# Visualize the patch embeddings before training.
patches_viz = embeddings[0].detach().numpy()  # shape: [196, 768]

plt.figure(figsize=(15, 8))
plt.imshow(patches_viz, aspect="auto", cmap="viridis")
plt.colorbar()
plt.title("Visualization of All Patch Embeddings")
plt.xlabel("Embedding Dimension")
plt.ylabel("Patch Number")
plt.show()

# Get the patch embeddings from the Hugging Face vision model for comparison.
vision_model.eval()
input = processor(images=img, return_tensors="pt")

with torch.no_grad():
    patch_embeddings = vision_model.vision_model.embeddings(input.pixel_values)

print(patch_embeddings.shape)

# Visualize the patch embeddings from the pretrained model.
patches_viz = patch_embeddings[0].detach().numpy()  # shape: [196, 768]

plt.figure(figsize=(15, 8))
plt.imshow(patches_viz, aspect="auto", cmap="viridis")
plt.colorbar()
plt.title("Trained Model: All Patch Embeddings")
plt.xlabel("Embedding Dimension")
plt.ylabel("Patch Number")
plt.show()

# ---------------------------
# Siglip Vision Embeddings Module
# ---------------------------

@dataclass
class SiglipVisionConfig:
    """
    Configuration class for Siglip vision models.
    """
    num_channels: int = 3
    embed_dim: int = 768
    image_size: int = 224
    patch_size: int = 16

class SiglipVisionEmbeddings(nn.Module):
    """
    Module that generates patch embeddings and adds positional embeddings.
    """
    def __init__(self, config: SiglipVisionConfig):
        """
        Initialize the SiglipVisionEmbeddings module.

        Args:
            config (SiglipVisionConfig): Configuration parameters for the embeddings.
        """
        super().__init__()
        self.config = config
        self.num_channels = config.num_channels
        self.embed_dim = config.embed_dim
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=self.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        """
        Forward pass to compute patch embeddings with positional encoding.

        Args:
            pixel_values (torch.FloatTensor): Input images of shape (B, C, H, W).

        Returns:
            torch.Tensor: Embeddings with shape (B, num_patches, embed_dim).
        """
        B, C, H, W = pixel_values.shape  # batch, channel, height, width
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(start_dim=2, end_dim=-1)
        embeddings = embeddings.transpose(1, 2)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings

# Instantiate and test the SiglipVisionEmbeddings module.
embd = SiglipVisionEmbeddings(SiglipVisionConfig())
print(embd(image_tensor).shape)

# ---------------------------
# Loading Pretrained Weights into Embeddings
# ---------------------------

# Load state dictionaries from our module and the HF vision model.
our_state_dict = embd.state_dict()
hf_state_dict = {
    k.replace("vision_model.embeddings.", ""): v
    for k, v in vision_model.state_dict().items() if "vision_model.embeddings." in k
}
our_state_dict.update(hf_state_dict)
embd.load_state_dict(our_state_dict)

with torch.no_grad():
    our_output = embd(image_tensor)
    hf_output = vision_model.vision_model.embeddings(image_tensor)
    print("Max difference between our output and HF output:",
          torch.max(torch.abs(our_output - hf_output)))  # Expected to be 0

# ---------------------------
# Attention Modules
# ---------------------------

class Head(nn.Module):
    """
    A single attention head used in multi-head attention.
    """
    def __init__(self, n_in, n_head, context_length):
        """
        Initialize a single head of multi-head attention.

        Args:
            n_in (int): Input feature dimension.
            n_head (int): Dimension of the head.
            context_length (int): Context length (not used directly here).
        """
        super().__init__()
        self.head_size = n_head
        self.key = nn.Linear(n_in, n_head, bias=False)
        self.query = nn.Linear(n_in, n_head, bias=False)
        self.value = nn.Linear(n_in, n_head, bias=False)

    def forward(self, x):
        """
        Forward pass for the attention head.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).

        Returns:
            torch.Tensor: Output tensor after attention of shape (B, T, n_head).
        """
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))
        wei = F.softmax(wei, dim=-1)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention that concatenates the outputs of all heads.
    """
    def __init__(self, num_head, n_in, head_size, context_length):
        """
        Initialize the multi-head attention module.

        Args:
            num_head (int): Number of attention heads.
            n_in (int): Input feature dimension.
            head_size (int): Dimension of each head.
            context_length (int): Context length (for head initialization).
        """
        super().__init__()
        self.head_size = head_size
        self.num_head = num_head
        self.heads = [Head(n_in, head_size, context_length) for _ in range(num_head)]
        self.proj = nn.Linear(n_in, n_in)

    def forward(self, x):
        """
        Forward pass for multi-head attention.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).

        Returns:
            torch.Tensor: Output tensor after multi-head attention of shape (B, T, C).
        """
        out = [h(x) for h in self.heads]
        out = torch.concat(out, -1)
        out = self.proj(out)
        return out

# Test multi-head attention.
num_attention_heads = 12
hidden_size = 768
attn = MultiHeadAttention(num_head=12, n_in=768, head_size=64, context_length=196)
print(attn(torch.randn(1, 196, 768)))

# ---------------------------
# Siglip Attention Module with Pretrained Weight Transfer
# ---------------------------

@dataclass
class SiglipVisionConfig:
    """
    Extended configuration for Siglip vision models including attention parameters.
    """
    num_channels: int = 3
    image_size: int = 224
    patch_size: int = 16
    num_attention_heads: int = 12
    hidden_size: int = 768
    attention_dropout: float = 0.0

class SiglipAttention(nn.Module):
    """
    Self-attention module for the Siglip vision transformer.
    """
    def __init__(self, config: SiglipVisionConfig):
        """
        Initialize the attention module.

        Args:
            config (SiglipVisionConfig): Configuration with attention and model parameters.
        """
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states):
        """
        Forward pass for the attention module.

        Args:
            hidden_states (torch.Tensor): Input embeddings with shape (B, T, embed_dim).

        Returns:
            torch.Tensor: Output after applying attention, shape (B, T, embed_dim).
        """
        B, T, C = hidden_states.shape
        q_states = self.q_proj(hidden_states)
        k_states = self.k_proj(hidden_states)
        v_states = self.v_proj(hidden_states)

        # Reshape to (B, num_heads, T, head_dim)
        q_states = q_states.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        k_states = k_states.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v_states = v_states.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)

        # Compute attention weights and apply scaling.
        attn_weights = (q_states @ k_states.transpose(-2, -1)) * (1.0 / math.sqrt(k_states.size(-1)))
        attn_weights = F.softmax(attn_weights, dim=-1).to(q_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_outs = attn_weights @ v_states

        # Reshape back to (B, T, C)
        attn_outs = attn_outs.transpose(1, 2).reshape(B, T, C).contiguous()
        attn_outs = self.out_proj(attn_outs)
        return attn_outs

# Test the SiglipAttention module.
batch_size = 1
num_patches = 196
embed_dim = 768
hidden_states = torch.randn(batch_size, num_patches, embed_dim)
config = SiglipVisionConfig(
    attention_dropout=0.0,
    num_attention_heads=12,
    hidden_size=768
)
attention = SiglipAttention(config)
output = attention(hidden_states)

print(f"Input shape: {hidden_states.shape}")
print(f"Output shape: {output.shape}")

# Transfer pretrained weights from Hugging Face model to our attention module.
hf_state_dict = vision_model.vision_model.state_dict()
our_state_dict = attention.state_dict()

key_mapping = {
    'k_proj.weight': 'encoder.layers.0.self_attn.k_proj.weight',
    'k_proj.bias': 'encoder.layers.0.self_attn.k_proj.bias',
    'v_proj.weight': 'encoder.layers.0.self_attn.v_proj.weight',
    'v_proj.bias': 'encoder.layers.0.self_attn.v_proj.bias',
    'q_proj.weight': 'encoder.layers.0.self_attn.q_proj.weight',
    'q_proj.bias': 'encoder.layers.0.self_attn.q_proj.bias',
    'out_proj.weight': 'encoder.layers.0.self_attn.out_proj.weight',
    'out_proj.bias': 'encoder.layers.0.self_attn.out_proj.bias'
}

for our_key, hf_key in key_mapping.items():
    our_state_dict[our_key].copy_(hf_state_dict[hf_key])

print(attention.load_state_dict(our_state_dict))

with torch.no_grad():
    our_output = attention(hidden_states)
    hf_output = vision_model.vision_model.encoder.layers[0].self_attn(hidden_states)[0]
    max_diff = torch.max(torch.abs(our_output - hf_output))
    print(f"Max difference between our output and HF output: {max_diff:.6f}")
    print((torch.isclose(our_output, hf_output, atol=1e-6)==0).sum())

# ---------------------------
# MLP Module for the Transformer
# ---------------------------

@dataclass
class SiglipVisionConfig:
    """
    Extended configuration for the vision transformer including MLP parameters.
    """
    num_channels: int = 3
    image_size: int = 224
    patch_size: int = 16
    num_attention_heads: int = 12
    hidden_size: int = 768  # Renamed from embed_dim.
    attention_dropout: float = 0.0
    intermediate_size: int = 3072

class SiglipMLP(nn.Module):
    """
    MLP block used in the transformer encoder layer.
    """
    def __init__(self, config: SiglipVisionConfig):
        """
        Initialize the MLP block.

        Args:
            config (SiglipVisionConfig): Configuration with model parameters.
        """
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP block.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape (B, T, hidden_size).

        Returns:
            torch.Tensor: Output tensor of shape (B, T, hidden_size).
        """
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        hidden_states = self.fc2(hidden_states)
        return hidden_states

mlp = SiglipMLP(SiglipVisionConfig(hidden_size=768, intermediate_size=3072))
print(mlp(torch.randn(1, 196, 768)).shape)

# ---------------------------
# Siglip Vision Embeddings with LayerNorm
# ---------------------------

@dataclass
class SiglipVisionConfig:
    """
    Configuration for the vision transformer with layer normalization.
    """
    num_channels: int = 3
    image_size: int = 224
    patch_size: int = 16
    num_attention_heads: int = 12
    hidden_size: int = 768  # Renamed from embed_dim.
    attention_dropout: float = 0.0
    intermediate_size: int = 3072
    layer_norm_eps: float = 1e-6

class SiglipVisionEmbeddings(nn.Module):
    """
    Vision embeddings module that creates patch and positional embeddings.
    """
    def __init__(self, config: SiglipVisionConfig):
        """
        Initialize the vision embeddings module.

        Args:
            config (SiglipVisionConfig): Configuration for embeddings.
        """
        super().__init__()
        self.config = config
        self.num_channels = config.num_channels
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=self.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,  # No overlapping since stride equals kernel size.
            padding="valid",
        )
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches  # Number of positions equals number of patches.
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_patches).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        """
        Forward pass to compute patch embeddings with positional encoding.

        Args:
            pixel_values (torch.FloatTensor): Input images of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output embeddings of shape (B, num_patches, embed_dim).
        """
        B, C, H, W = pixel_values.shape
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(start_dim=2, end_dim=-1)
        embeddings = embeddings.transpose(1, 2)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings

# ---------------------------
# Transformer Encoder Layer
# ---------------------------

class SiglipEncoderLayer(nn.Module):
    """
    A single transformer encoder layer composed of self-attention, MLP, and layer normalization.
    """
    def __init__(self, config: SiglipVisionConfig):
        """
        Initialize the encoder layer.

        Args:
            config (SiglipVisionConfig): Configuration for the encoder layer.
        """
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        """
        Forward pass for the encoder layer.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape (B, T, hidden_size).

        Returns:
            torch.Tensor: Output tensor of shape (B, T, hidden_size).
        """
        # Self-attention block with residual connection.
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        # MLP block with residual connection.
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

encoder_layer = SiglipEncoderLayer(SiglipVisionConfig(hidden_size=768, intermediate_size=3072))
print(encoder_layer(torch.randn(1, 196, 768)).shape)

print(vision_model)

# ---------------------------
# Transformer Encoder
# ---------------------------

@dataclass
class SiglipVisionConfig:
    """
    Extended configuration for the full transformer encoder.
    """
    num_hidden_layers: int = 12  # Number of encoder layers.
    num_channels: int = 3
    image_size: int = 224
    patch_size: int = 16
    num_attention_heads: int = 12
    hidden_size: int = 768
    intermediate_size: int = 3072
    layer_norm_eps: float = 1e-6
    attention_dropout: float = 0.0

class SiglipEncoder(nn.Module):
    """
    Transformer encoder composed of multiple encoder layers.
    """
    def __init__(self, config: SiglipVisionConfig):
        """
        Initialize the encoder.

        Args:
            config (SiglipVisionConfig): Configuration for the transformer encoder.
        """
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states):
        """
        Forward pass for the transformer encoder.

        Args:
            hidden_states (torch.Tensor): Input embeddings of shape (B, T, hidden_size).

        Returns:
            torch.Tensor: Encoded representations of shape (B, T, hidden_size).
        """
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states

encoder = SiglipEncoder(SiglipVisionConfig(hidden_size=768, intermediate_size=3072))
print(encoder(torch.randn(1, 196, 768)).shape)

# ---------------------------
# Vision Transformer Assembly
# ---------------------------

class SiglipVisionTransformer(nn.Module):
    """
    Vision Transformer model that combines embeddings, encoder, and final layer normalization.
    """
    def __init__(self, config: SiglipVisionConfig):
        """
        Initialize the vision transformer.

        Args:
            config (SiglipVisionConfig): Configuration for the vision transformer.
        """
        super().__init__()
        self.config = config
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values):
        """
        Forward pass for the vision transformer.

        Args:
            pixel_values (torch.Tensor): Input images of shape (B, C, H, W).

        Returns:
            torch.Tensor: Final transformer output of shape (B, T, hidden_size).
        """
        hidden_states = self.embeddings(pixel_values)
        last_hidden_state = self.encoder(hidden_states)
        last_hidden_state = self.post_layernorm(last_hidden_state)
        return last_hidden_state

siglip = SiglipVisionTransformer(SiglipVisionConfig(hidden_size=768, intermediate_size=3072))
print(siglip(image_tensor).shape)

class SiglipVisionModel(nn.Module):
    """
    Wrapper model that contains the Siglip vision transformer.
    """
    def __init__(self, config: SiglipVisionConfig):
        """
        Initialize the SiglipVisionModel.

        Args:
            config (SiglipVisionConfig): Configuration for the vision model.
        """
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values):
        """
        Forward pass for the vision model.

        Args:
            pixel_values (torch.Tensor): Input images of shape (B, C, H, W).

        Returns:
            torch.Tensor: Vision model output.
        """
        return self.vision_model(pixel_values)

# Instantiate the complete vision model.
siglip = SiglipVisionModel(SiglipVisionConfig(hidden_size=768, intermediate_size=3072))
print(siglip(image_tensor).shape)

# ---------------------------
# Load Pretrained Weights from HF Vision Model
# ---------------------------

hf_state_dict = vision_model.state_dict()
our_state_dict = siglip.state_dict()

# Load the Hugging Face state dictionary into our model.
print(siglip.load_state_dict(hf_state_dict))
