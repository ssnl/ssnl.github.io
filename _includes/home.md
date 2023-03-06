I am a current PhD student at [MIT CSAIL](https://www.csail.mit.edu/) working with [Antonio Torralba](https://web.mit.edu/torralba/www/) and [Phillip Isola](https://web.mit.edu/phillipi/) on machine learning.
My current focuses are **geometric structures of learned representations**, and **enabling efficient, adaptive and general agents via such representations**. I am broadly interested in representation learning, reinforcement learning, synthetic training data, dataset distillation, and machine learning in general.
{: style="text-align: justify"}

I have spent time at Meta AI working with [Yuandong Tian](https://yuandong-tian.com/){:.color}, [Amy Zhang](https://amyzhang.github.io/), and [Simon S. Du](https://simonshaoleidu.com/). I also collaborate with [Alyosha Efros](https://people.eecs.berkeley.edu/~efros/) and [Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/).
{: style="text-align: justify"}

Before joining MIT, I worked at [Facebook AI Research (now Meta AI)](https://research.fb.com/category/facebook-ai-research-fair/) on [PyTorch](https://pytorch.org/), and studied computer science and statistics at UC Berkeley, where I was fortunate to work with [Stuart J. Russell](http://people.eecs.berkeley.edu/~russell/){:.color}, [Ren Ng](https://www2.eecs.berkeley.edu/Faculty/Homepages/yirenng.html){:.color}, and [Alyosha Efros](https://people.eecs.berkeley.edu/~efros/){:.color}.
{: style="text-align: justify;"}

Click [here](./assets/docs/about/cv.pdf) for my CV.

## Open Source Projects

1. [PyTorch](https://pytorch.org/) core developer (2017 - 2019, initial team size <10). Data loading, CUDA/CPU kernels, autograd optimization, ML ops, Python binding, etc.
2. [`torchreparam`](https://github.com/ssnl/PyTorch-Reparam-Module) developer (2019 - 2020). One of the earliest PyTorch toolkits for re-parametrizing neural networks, e.g., for hyper-nets and meta-learning.
3. [`torchqmet`](https://github.com/quasimetric-learning/torch-quasimetric) developer (2022 - now). PyTorch toolkit for SOTA [quasimetric learning](./interval_quasimetric_embedding).
4. [CycleGAN and pix2pix in PyTorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) maintainer (2018 - now). 18.9k stars.

See below for open source code for my researches.

## Selected Publications

1. **Optimal Goal-Reaching Reinforcement Learning via Quasimetric Learning** <br />
    **Tongzhou Wang**, Antonio Torralba, Phillip Isola, Amy Zhang
    {: style="margin-bottom: 0"}

1. **Improved Representation of Asymmetrical Distances with Interval Quasimetric Embeddings**<br />[[<span class="small__tt">**NeurIPS 2022 NeurReps Workshop**{: .colorful}</span>](https://www.neurreps.org/)] [[Project Page](./interval_quasimetric_embedding){: .small__tt}] [[arXiv](https://arxiv.org/abs/2211.15120){: .small__tt}] [[PyTorch Package for Quasimetric Learning](https://github.com/quasimetric-learning/torch-quasimetric){: .small__tt}] <br />
    **Tongzhou Wang**, Phillip Isola    
    {: style="margin-bottom: 0"}

    <img src="./interval_quasimetric_embedding/images/iqe_compute_nobg.png" alt="computing-iqe" style="width:97%;margin-top:2.5%">

1. **Denoised MDPs: Learning World Models Better Than The World**<br />[<span class="small__tt">**ICML 2022**{: .colorful}</span>] [[Project Page](./denoised_mdp){: .small__tt}] [[arXiv](https://arxiv.org/abs/2206.15477){: .small__tt}] [[code](https://github.com/facebookresearch/denoised_mdp){: .small__tt}] <br />
    **Tongzhou Wang**, Simon S. Du, Antonio Torralba, Phillip Isola, Amy Zhang, Yuandong Tian
    {: style="margin-bottom: 0"}

    <video src="https://user-images.githubusercontent.com/5674597/173155667-d4bcc7af-1f12-4ba3-a733-ef9d5f631c96.mp4" controls="controls" autoplay="true" loop="true" muted="muted" class="d-block rounded-bottom-2 border-top width-fit" style="width:100%">  </video>

2. **On the Learning and Learnability of Quasimetrics**<br />[<span class="small__tt">**ICLR 2022**{: .colorful}</span>] [[Project Page](/quasimetric){: .small__tt}] [[arXiv](https://arxiv.org/abs/2206.15478){: .small__tt}] [[OpenReview](https://openreview.net/forum?id=y0VvIg25yk){: .small__tt}] [[code](https://github.com/SsnL/poisson_quasimetric_embedding){: .small__tt}] <br />
    **Tongzhou Wang**, Phillip Isola
    {: style="margin-bottom: 0"}

    <img src="./quasimetric/images/function_spaces_cropped.png" alt="quasimetric-function-spaces" style="width:100%">

3. **Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere**<br />[<span class="small__tt">**ICML 2020**{: .colorful}</span>] [[Project Page](/hypersphere){: .small__tt}] [[arXiv](https://arxiv.org/abs/2005.10242){: .small__tt}] [[code](https://github.com/SsnL/align_uniform){: .small__tt}] <br />
    **Tongzhou Wang**, Phillip Isola

    <div style="display: flex; width: 100%;margin-top: -0.5em">
    <table style="width:100%; height: 210px">
        <tr>
        <td style="width:42%;border-bottom: 0px;padding:0px;vertical-align: bottom;text-align: left">
            <img src="/assets/images/hypersphere_stl10_scatter_linear_output.png" alt="hypersphere_stl10_scatter_linear_output" />
        </td>
        <td style="width:1%;border-bottom: 0px;padding:0px;" />
        <td style="width:57%;border-bottom: 0px;padding-bottom:1.03em;padding-left:0px;padding-right:0px;text-align:right;vertical-align: bottom;">
            <div style="font-size: 0.735em;display: inline-block;text-align:left;width:100%">
              <div style="background: #ffffff; overflow:auto;width:auto;border:solid gray;border-width:.1em .1em .1em .8em;padding:.2em .6em">
        <pre style="margin: 0; line-height: 160%">
<span style="color: #888888"># bsz : batch size (number of positive pairs)</span>
<span style="color: #888888"># d   : latent dim</span>
<span style="color: #888888"># x   : Tensor, shape=[bsz, d]</span>
<span style="color: #888888">#       latents for one side of positive pairs</span>
<span style="color: #888888"># y   : Tensor, shape=[bsz, d]</span>
<span style="color: #888888">#       latents for the other side of positive pairs</span>
<span style="color: #008800; font-weight: bold">def</span> <span style="color: #0066BB; font-weight: bold">align_loss</span>(x, y, alpha<span style="color: #333333">=</span><span style="color: #0000DD; font-weight: bold">2</span>):
<span style="color: #008800; font-weight: bold">    return</span> (x <span style="color: #333333">-</span> y)<span style="color: #333333">.</span>norm(p<span style="color: #333333">=</span><span style="color: #0000DD; font-weight: bold">2</span>, dim<span style="color: #333333">=</span><span style="color: #0000DD; font-weight: bold">1</span>)<span style="color: #333333">.</span>pow(alpha)<span style="color: #333333">.</span>mean()<br/>
<span style="color: #008800; font-weight: bold">def</span> <span style="color: #0066BB; font-weight: bold">uniform_loss</span>(x, t<span style="color: #333333">=</span><span style="color: #0000DD; font-weight: bold">2</span>):
<span style="color: #008800; font-weight: bold">    return</span> torch<span style="color: #333333">.</span>pdist(x, p<span style="color: #333333">=</span><span style="color: #0000DD; font-weight: bold">2</span>)<span style="color: #333333">.</span>pow(<span style="color: #0000DD; font-weight: bold">2</span>)<span style="color: #333333">.</span>mul(<span style="color: #333333">-</span>t)<span style="color: #333333">.</span>exp()<span style="color: #333333">.</span>mean()<span style="color: #333333">.</span>log()</pre>
      </div>
              <div style="text-align: center; font-size: 1.35em"><a href='https://github.com/SsnL/align_uniform'>PyTorch implementation</a> of the alignment and uniformity losses</div>
            </div>
        </td>
      </tr>
    </table>
    </div>

4. **Dataset Distillation**<br />[[Project Page](/dataset_distillation){: .small__tt}] [[arXiv](https://arxiv.org/abs/1811.10959){: .small__tt}] [[code](https://github.com/SsnL/dataset-distillation){: .small__tt}] <br />
    **Tongzhou Wang**, Jun-Yan Zhu, Antonio Torralba, Alexei A. Efros

    ![dataset_distillation_fixed_mnist](/assets/images/dataset_distillation_fixed_mnist.png){: style="width:100%;"}

5. **Meta-Learning MCMC Proposals**<br />[<span class="small__tt">**NeurIPS 2018**{: .colorful}</span>] [<span class="small__tt">**PROBPROG 2018**{: .colorful}</span>] [[ICML 2017 AutoML Workshop Oral](./automl_17/slides.pdf)] [[arXiv](https://arxiv.org/abs/1708.06040){: .small__tt}] <br />
    **Tongzhou Wang**, Yi Wu, David A. Moore, Stuart J. Russell

    ![meta_learning_mcmc_gmm_trace](/assets/images/meta_learning_mcmc_gmm_trace.png){: style="width:100%;"}

6. **Learning to Synthesize a 4D RGBD Light Field from a Single Image**<br />[<span class="small__tt">**ICCV 2017**{: .colorful}</span>] [[arXiv](https://arxiv.org/abs/1708.03292){: .small__tt}] <br />
    Pratul Srinivasan, **Tongzhou Wang**, Ashwin Sreelal, Ravi Ramamoorthi, Ren Ng
    ![light-field-synthesis-pipeline](/assets/images/2d_to_4d_pipeline.png){: style="width:100%;"}

<!--
## Selected Projects

1. **Improved Training of Cycle-Consistent Adversarial Networks**

    **Tongzhou Wang** and Yihan Lin with research group of Prof. Alexei A. Efros

    Related report: [**CycleGAN with Better Cycles**{: style="font-size: 0.95em"}](/better_cycles/report.pdf).

2. **Modeling Punctuations in Online Reviews**<br/>[[technical report](/punctuations/report.pdf){: .small__tt}]

    **Tongzhou Wang**

    ![light-field-synthesis-pipeline](/assets/images/punctuation_neg_ex.png){: style="max-height:7em;width:auto;"}
-->

