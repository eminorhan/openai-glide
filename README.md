# GLIDE

This is a clone of OpenAI's GLIDE repo for experimentation.

Paper: [GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models](https://arxiv.org/abs/2112.10741).

For details on the pre-trained models in this repository, see the [Model Card](model-card.md).

For detailed usage examples, see the [notebooks](notebooks) directory.

 * The [text2im](notebooks/text2im.ipynb) [![][colab]][colab-text2im] notebook shows how to use GLIDE (filtered) with classifier-free guidance to produce images conditioned on text prompts. 
 * The [inpaint](notebooks/inpaint.ipynb) [![][colab]][colab-inpaint] notebook shows how to use GLIDE (filtered) to fill in a masked region of an image, conditioned on a text prompt. 
 * The [clip_guided](notebooks/clip_guided.ipynb) [![][colab]][colab-guided] notebook shows how to use GLIDE (filtered) + a filtered noise-aware CLIP model to produce images conditioned on text prompts. 

[colab]: <https://colab.research.google.com/assets/colab-badge.svg>
[colab-text2im]: <https://colab.research.google.com/github/openai/glide-text2im/blob/main/notebooks/text2im.ipynb>
[colab-inpaint]: <https://colab.research.google.com/github/openai/glide-text2im/blob/main/notebooks/inpaint.ipynb>
[colab-guided]: <https://colab.research.google.com/github/openai/glide-text2im/blob/main/notebooks/clip_guided.ipynb>
