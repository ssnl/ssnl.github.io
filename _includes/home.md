I am currently working at [Facebook AI Research (FAIR)](https://research.fb.com/category/facebook-ai-research-fair/). Before joining Facebook, I studied computer science and statistics at University of California, Berkeley. Since my undergraduate study, I have worked with Professor [Stuart J. Russell](http://people.eecs.berkeley.edu/~russell/), Professor [Ren Ng](https://www2.eecs.berkeley.edu/Faculty/Homepages/yirenng.html){:.color}, and Professor [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros/) as a researcher at [Berkeley AI Research (BAIR)](http://bair.berkeley.edu/).
{: style="text-align: justify;"}

Please find my full résumé [here](/assets/docs/about/resume.pdf).

## Publications

1. **Dataset Distillation**<br />[[Project Page](/dataset_distillation){: .small__tt}] [[code](https://github.com/SsnL/dataset-distillation){: .small__tt}] [[arXiv](https://arxiv.org/abs/1811.10959){: .small__tt}]

    Tongzhou Wang, Jun-Yan Zhu, Antonio Torralba, Alexei A. Efros

    We attempt to distill the knowledge from a large training dataset into a small one. The idea is to <em>synthesize</em> a small number of data points that do not need to come from the correct data distribution, but will, when given to the learning algorithm as training data, approximate the model trained on the original data. For example, we show that it is possible to compress 60,000 MNIST training images into just 10 synthetic <em style="text-decoration: underline;">distilled images</em> and achieve close to original performance with only a few steps of gradient descent, given a fixed network initialization. Experiments on multiple datasets show the advantage of our approach compared to alternative methods in various initialization settings and with different learning objectives.
    {: style="text-align: justify;"}

    ![dataset_distillation_fixed_mnist](/assets/images/dataset_distillation_fixed_mnist.png){: style="max-height:12.5em;width:auto;"}

2. **Meta-Learning MCMC Proposals**<br />[<span class="small__tt">**NIPS 2018**{: .colorful}</span>] [[arXiv](https://arxiv.org/abs/1708.06040){: .small__tt}]

    Tongzhou Wang, Yi Wu, David A. Moore, Stuart J. Russell

    Automated MCMC proposal construction by training neural networks as fast approximations to block Gibbs conditionals. The learned proposals generalize to occurrences of common structural motifs both within a given model and across models, allowing for the construction of a library of learned inference primitives that can accelerate inference on unseen models with no model-specific training required.
    {: style="text-align: justify;"}

    [Oral presentation](/automl_17/slides.pdf) at ICML 2017 AutoML workshop.

    ![meta_learning_mcmc_gmm_trace](/assets/images/meta_learning_mcmc_gmm_trace.png){: style="max-height:12.5em;width:auto;"}

3. **Learning to Synthesize a 4D RGBD Light Field from a Single Image**<br />[<span class="small__tt">**ICCV 2017**{: .colorful}</span>] [[arXiv](https://arxiv.org/abs/1708.03292){: .small__tt}]

    Pratul Srinivasan, Tongzhou Wang, Ashwin Sreelal, Ravi Ramamoorthi, Ren Ng

    A machine learning algorithm that takes as input a 2D RGB image and synthesizes a 4D RGBD light field (color and depth of the scene in each ray direction). For training, we introduce the largest public light field dataset. Our algorithm is unique in predicting RGBD for each light field ray and improving unsupervised single image depth estimation by enforcing consistency of ray depths that should intersect the same scene point.
    {: style="text-align: justify;"}

    ![light-field-synthesis-pipeline](/assets/images/2d_to_4d_pipeline.png){: style="max-height:12.5em;width:auto;"}

## Research Projects

1. **Improved Training of Cycle-Consistent Adversarial Networks**

    Tongzhou Wang with research group of Prof. Alexei A. Efros

    On-going project on improving CycleGAN by designing better formulation and/or automatic dataset selection algorithms.
    {: style="text-align: justify;"}

    Relevant vision course project: **CycleGAN with Better Cycles**{: style="font-size: 0.95em"} [[paper](/better_cycles/report.pdf){: .small__tt}, [slides](/better_cycles/slides.pdf){: .small__tt}].

2. **Analysis on Punctuations in Online Reviews** [[paper](/punctuations/report.pdf){: .small__tt}, [poster](/punctuations/poster.pdf){: .small__tt}]

    Tongzhou Wang

    Analysis on punctuation structures in positive and negative online Steam reviews with an HMM model where the auxiliary sentence type variables are hidden and conditional probabilities of observed punctuations are modeled as from Markov chains based on the sentence types.
    {: style="text-align: justify;"}

    Course project of graduate-level statistical learning theory class.

    ![light-field-synthesis-pipeline](/assets/images/punctuation_neg_ex.png){: style="max-height:7em;width:auto;"}
