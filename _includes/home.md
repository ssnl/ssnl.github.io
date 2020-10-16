I am a current PhD student at [MIT CSAIL](https://www.csail.mit.edu/) working with [Antonio Torralba](https://web.mit.edu/torralba/www/) and [Phillip Isola](https://web.mit.edu/phillipi/). Previously, I worked at [Facebook AI Research (FAIR)](https://research.fb.com/category/facebook-ai-research-fair/) on [PyTorch](https://pytorch.org/), and studied computer science and statistics at University of California, Berkeley, where I was fortunate to work with [Stuart J. Russell](http://people.eecs.berkeley.edu/~russell/){:.color}, [Ren Ng](https://www2.eecs.berkeley.edu/Faculty/Homepages/yirenng.html){:.color}, and [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros/){:.color} at [Berkeley AI Research (BAIR)](http://bair.berkeley.edu/).
{: style="text-align: justify;"}

I aim to understand when and why machine learning models work well/badly, and build better and more generalizable AI systems. Towards this goal, questions I think about are often related to generalization, distribution shift, quantifying complexity, learning dynamics, etc. My current areas of focus include representation learning, reinforcement learning, and generative modeling.  

Please find my CV [here](/assets/docs/about/cv.pdf).

## Selected Publications

1. **Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere**<br />[<span class="small__tt">**ICML 2020**{: .colorful}</span>] [[Project Page](/hypersphere){: .small__tt}] [[code](https://github.com/SsnL/align_uniform){: .small__tt}] [[arXiv](https://arxiv.org/abs/2005.10242){: .small__tt}]

    Tongzhou Wang, Phillip Isola

    <p style="margin-bottom: 0;text-align: justify;">Contrastive representation learning has been outstandingly successful in practice. In this work, we identify two key properties related to the contrastive loss:</p>

    1. <div style="margin-bottom: 0;text-align: justify;"><em style="text-decoration: underline;">alignment</em> (closeness) of features from positive pairs, and</div>

    2. <div style="margin-bottom: 0;text-align: justify;"><em style="text-decoration: underline;">uniformity</em> of the induced distribution of the (normalized) features on the hypersphere.</div>

    We prove that, asymptotically, the contrastive loss optimizes these properties, and analyze their positive effects on downstream tasks. Empirically, we introduce an optimizable metric to quantify each property. Extensive experiments on standard vision and language datasets confirm the strong agreement between <em>both metrics</em> and downstream task performance. Remarkably, directly optimizing for these two metrics leads to representations with comparable or better performance at downstream tasks than contrastive learning.
    {: style="text-align: justify;"}

    <div style="display: flex; width: 100%;margin-top: -0.5em">
    <img src="/assets/images/hypersphere_stl10_scatter_linear_output.png" alt="hypersphere_stl10_scatter_linear_output" style="width:40%;height:auto;">
    <div style="max-width: 62%;text-align: center;font-size: 0.46em;display: inline-block;text-align:left; padding-top: 2.3em; right: 0px; position: absolute;">
      <div style="background: #ffffff; overflow:auto;width:auto;border:solid gray;border-width:.1em .1em .1em .8em;padding:.2em .6em">
        <pre style="margin: 0; line-height: 160%"><span style="color: #888888"># bsz : batch size (number of positive pairs)</span>
    <span style="color: #888888"># d   : latent dim</span>
    <span style="color: #888888"># x   : Tensor, shape=[bsz, d]</span>
    <span style="color: #888888">#       latents for one side of positive pairs</span>
    <span style="color: #888888"># y   : Tensor, shape=[bsz, d]</span>
    <span style="color: #888888">#       latents for the other side of positive pairs</span>
    <span style="color: #008800; font-weight: bold">def</span> <span style="color: #0066BB; font-weight: bold">align_loss</span>(x, y, alpha<span style="color: #333333">=</span><span style="color: #0000DD; font-weight: bold">2</span>):
        <span style="color: #008800; font-weight: bold">return</span> (x <span style="color: #333333">-</span> y)<span style="color: #333333">.</span>norm(p<span style="color: #333333">=</span><span style="color: #0000DD; font-weight: bold">2</span>, dim<span style="color: #333333">=</span><span style="color: #0000DD; font-weight: bold">1</span>)<span style="color: #333333">.</span>pow(alpha)<span style="color: #333333">.</span>mean()

    <span style="color: #008800; font-weight: bold">def</span> <span style="color: #0066BB; font-weight: bold">uniform_loss</span>(x, t<span style="color: #333333">=</span><span style="color: #0000DD; font-weight: bold">2</span>):
        <span style="color: #008800; font-weight: bold">return</span> torch<span style="color: #333333">.</span>pdist(x, p<span style="color: #333333">=</span><span style="color: #0000DD; font-weight: bold">2</span>)<span style="color: #333333">.</span>pow(<span style="color: #0000DD; font-weight: bold">2</span>)<span style="color: #333333">.</span>mul(<span style="color: #333333">-</span>t)<span style="color: #333333">.</span>exp()<span style="color: #333333">.</span>mean()<span style="color: #333333">.</span>log()</pre>
      </div>
      <div style="text-align: center; font-size: 1.2em"><a href='https://github.com/SsnL/align_uniform'>PyTorch implementation</a> of the alignment and uniformity losses</div>
    </div>
    </div>

2. **Dataset Distillation**<br />[[Project Page](/dataset_distillation){: .small__tt}] [[code](https://github.com/SsnL/dataset-distillation){: .small__tt}] [[arXiv](https://arxiv.org/abs/1811.10959){: .small__tt}]

    Tongzhou Wang, Jun-Yan Zhu, Antonio Torralba, Alexei A. Efros

    We attempt to distill the knowledge from a large training dataset into a small one. The idea is to <em>synthesize</em> a small number of data points that do not need to come from the correct data distribution, but will, when given to the learning algorithm as training data, approximate the model trained on the original data. For example, we show that it is possible to compress 60,000 MNIST training images into just 10 synthetic <em style="text-decoration: underline;">distilled images</em> and achieve close to original performance with only a few steps of gradient descent, given a fixed network initialization. Experiments on multiple datasets show the advantage of our approach compared to alternative methods in various initialization settings and with different learning objectives.
    {: style="text-align: justify;"}

    ![dataset_distillation_fixed_mnist](/assets/images/dataset_distillation_fixed_mnist.png){: style="max-height:12.5em;width:auto;"}

3. **Meta-Learning MCMC Proposals**<br />[<span class="small__tt">**NIPS 2018**{: .colorful}</span>] [[arXiv](https://arxiv.org/abs/1708.06040){: .small__tt}]

    Tongzhou Wang, Yi Wu, David A. Moore, Stuart J. Russell

    Automated MCMC proposal construction by training neural networks as fast approximations to block Gibbs conditionals. The learned proposals generalize to occurrences of common structural motifs both within a given model and across models, allowing for the construction of a library of learned inference primitives that can accelerate inference on unseen models with no model-specific training required.
    {: style="text-align: justify;"}

    [Oral presentation](/automl_17/slides.pdf) at ICML 2017 AutoML workshop.

    ![meta_learning_mcmc_gmm_trace](/assets/images/meta_learning_mcmc_gmm_trace.png){: style="max-height:12.5em;width:auto;"}

4. **Learning to Synthesize a 4D RGBD Light Field from a Single Image**<br />[<span class="small__tt">**ICCV 2017**{: .colorful}</span>] [[arXiv](https://arxiv.org/abs/1708.03292){: .small__tt}]

    Pratul Srinivasan, Tongzhou Wang, Ashwin Sreelal, Ravi Ramamoorthi, Ren Ng

    A machine learning algorithm that takes as input a 2D RGB image and synthesizes a 4D RGBD light field (color and depth of the scene in each ray direction). For training, we introduce the largest public light field dataset. Our algorithm is unique in predicting RGBD for each light field ray and improving unsupervised single image depth estimation by enforcing consistency of ray depths that should intersect the same scene point.
    {: style="text-align: justify;"}

    ![light-field-synthesis-pipeline](/assets/images/2d_to_4d_pipeline.png){: style="max-height:12.5em;width:auto;"}

## Selected Projects

1. **Improved Training of Cycle-Consistent Adversarial Networks**

    Tongzhou Wang and Yihan Lin with research group of Prof. Alexei A. Efros

    Improving CycleGAN by designing better formulation and/or automatic dataset selection algorithms.
    {: style="text-align: justify;"}

    Relevant vision course project: **CycleGAN with Better Cycles**{: style="font-size: 0.95em"} [[paper](/better_cycles/report.pdf){: .small__tt}, [slides](/better_cycles/slides.pdf){: .small__tt}].

2. **Modeling Punctuations in Online Reviews** [[paper](/punctuations/report.pdf){: .small__tt}, [poster](/punctuations/poster.pdf){: .small__tt}]

    Tongzhou Wang

    Analysis on punctuation structures in positive and negative online Steam reviews with an HMM model where the auxiliary sentence type variables are hidden and conditional probabilities of observed punctuations are modeled as from Markov chains based on the sentence types.
    {: style="text-align: justify;"}

    Course project of graduate-level statistical learning theory class.

    ![light-field-synthesis-pipeline](/assets/images/punctuation_neg_ex.png){: style="max-height:7em;width:auto;"}
