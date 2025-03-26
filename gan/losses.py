import torch
from torch.nn.functional import binary_cross_entropy_with_logits


def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss.

    Uses the stable torch.nn.functional.binary_cross_entropy_with_logits
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.

    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """

    loss = None
    fake_loss = binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake), reduction='mean')
    real_loss = binary_cross_entropy_with_logits(logits_real, torch.ones_like(logits_real), reduction='mean')
    loss = fake_loss + real_loss

    return loss


def generator_loss(logits_fake):
    """
    Computes the generator loss.

    Uses the stable torch.nn.functional.binary_cross_entropy_with_logits
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """

    loss = None
    target = torch.ones_like(logits_fake)
    loss = binary_cross_entropy_with_logits(logits_fake, target, reduction='mean')

    return loss


def ls_discriminator_loss(scores_real, scores_fake):
    """
    Computes the Least-Squares GAN loss for the discriminator.

    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """

    loss = None

    loss = 0.5*(
      torch.nn.functional.mse_loss(torch.ones_like(scores_real), scores_real)+
      torch.nn.functional.mse_loss(torch.zeros_like(scores_fake), scores_fake))

    return loss


def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.

    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """

    loss = None

    target  = torch.ones_like(scores_fake)
    loss = 0.5*torch.nn.functional.mse_loss(target, scores_fake)

    return loss

def wasserstein_discriminator_loss(score_real, score_fake):
  loss_real = score_real.mean()
  loss_fake = score_fake.mean()
  loss = -(loss_real - loss_fake)
  return loss

def wasserstein_generator_loss(score_fake):
  return -score_fake.mean()

