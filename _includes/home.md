I am a machine learning PhD student at [MIT CSAIL](https://www.csail.mit.edu/) with [Antonio Torralba](https://web.mit.edu/torralba/www/) and [Phillip Isola](https://web.mit.edu/phillipi/).
My research focuses on the **structures of learned intelligence:**
{: style="text-align: justify; margin-bottom:3px"}
 + Understand how learning algorithms rely on structures/signals in data to produce models.
   <p style="margin-bottom:-7px"></p>
 + Improve efficiency and generality of learned perception & reasoning by incorporating new useful structures.
{: style="text-align: justify;padding-left:30px;margin-top:0px;margin-bottom:3px;font-size:0.885em"}

Broadly, I am interested in representation learning, reinforcement learning, synthetic data, and [dataset distillation](./dataset_distillation/).
{: style="text-align: justify"}

During PhD, I have spent time at Meta AI working with [Yuandong Tian](https://yuandong-tian.com/){:.color}, [Amy Zhang](https://amyzhang.github.io/), and [Simon S. Du](https://simonshaoleidu.com/). I also collaborate with [Alyosha Efros](https://people.eecs.berkeley.edu/~efros/) and [Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/).
{: style="text-align: justify"}

Before MIT, I was an early member of the [PyTorch](https://pytorch.org/) core team at [Facebook AI Research (now Meta AI)](https://research.fb.com/category/facebook-ai-research-fair/) (2017-2019). I completed my undergradute study at UC Berkeley (2013-2017), where I started my research with [Stuart  Russell](http://people.eecs.berkeley.edu/~russell/){:.color}, [Ren Ng](https://www2.eecs.berkeley.edu/Faculty/Homepages/yirenng.html){:.color}, and [Alyosha Efros](https://people.eecs.berkeley.edu/~efros/){:.color} on probabilistic inference, graphics, and image generative models.
{: style="text-align: justify;"}

At MIT, I helped develop the [6.S898 Deep Learning](https://phillipi.github.io/6.s898/) course, and served as the head TA.

Click [here](./assets/docs/about/cv.pdf) for my CV.

## Selected Open Source Projects  <a style="margin-left:7px; vertical-align:middle" href="https://github.com/ssnl">![GitHub User's stars](https://img.shields.io/github/stars/ssnl?affiliations=OWNER%2CCOLLABORATOR&logo=github&label=%40ssnl%20Stars){: style="height: 0.87em;vertical-align:baseline"}</a>

1. **[PyTorch](https://pytorch.org/) core developer (![v020](https://img.shields.io/badge/ver.-0.2.0-yellowgreen) 2017 - ![v100](https://img.shields.io/badge/ver.-1.0.0-red) 2019; team size <10)** <a style="margin-left:5px;vertical-align:text-bottom" href="https://github.com/pytorch/pytorch">![GitHub Repo stars](https://img.shields.io/github/stars/pytorch/pytorch?style=social)</a><br/>
    Data loading, CUDA/CPU kernels, ML ops, API design, autograd optimization, Python binding, etc. 
    {: style="margin-bottom: 0"}

2. **[CycleGAN and pix2pix in PyTorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) maintainer (2018 - now)** <a style="margin-left:5px;vertical-align:text-bottom" href="https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix">![GitHub Repo stars](https://img.shields.io/github/stars/junyanz/pytorch-CycleGAN-and-pix2pix?style=social)</a>
    {: style="margin-bottom: 0"}

2. **[`torchreparam`](https://github.com/ssnl/PyTorch-Reparam-Module) developer (2019 - 2020)** <a style="margin-left:5px;vertical-align:text-bottom" href="https://github.com/ssnl/PyTorch-Reparam-Module">![GitHub Repo stars](https://img.shields.io/github/stars/ssnl/PyTorch-Reparam-Module?style=social)</a><br/>
    One of the earliest PyTorch toolkits to **re-parametrize** neural nets (e.g., for hyper-nets and meta-learning).
    {: style="margin-bottom: 0"}

4. **[`Awesome-Dataset-Distillation`](https://github.com/Guang000/Awesome-Dataset-Distillation) maintainer (2022 - now)** <a style="margin-left:5px;vertical-align:text-bottom" href="https://github.com/Guang000/Awesome-Dataset-Distillation">![GitHub Repo stars](https://img.shields.io/github/stars/Guang000/Awesome-Dataset-Distillation?style=social)</a><br/> 
    Collection of [Dataset Distillation](./dataset_distillation) papers in machine learning and vision conferences.
    {: style="margin-bottom: 0em"}

3. **[`torchqmet`](https://github.com/quasimetric-learning/torch-quasimetric) developer (2022 - now)** <a style="margin-left:5px;vertical-align:text-bottom" href="https://github.com/quasimetric-learning/torch-quasimetric">![GitHub Repo stars](https://img.shields.io/github/stars/quasimetric-learning/torch-quasimetric?style=social)</a><br/> 
    PyTorch toolkit for SOTA [quasimetric](./interval_quasimetric_embedding) [learning](./quasimetric).
    {: style="margin-bottom: -0.65em"}

See below for open source code for my researches.
{: style="font-size:0.75em;margin-bottom: -0.35em"}

## Selected Publications <span style="margin-left:6px;font-size:0.8em">([full list<i class="ai fa-fw ai-google-scholar-square" aria-hidden="true" />](https://scholar.google.com/citations?user=14HASnUAAAAJ))</span>

1. **Optimal Goal-Reaching Reinforcement Learning via Quasimetric Learning**<br />[<span class="small__tt">**ICML 2023**{: .colorful}</span>][[Project Page](./quasimetric_rl){: .small__tt}] [[arXiv](https://arxiv.org/abs/2304.01203){: .small__tt}] [<span>Code Coming Soon</span>{: .small__tt}] <br />
    **Tongzhou Wang**, Antonio Torralba, Phillip Isola, Amy Zhang
    {: style="margin-bottom: 0"}

    <div class="table-like" style="justify-content:space-evenly;max-width:100%;width:100%;margin:auto;margin-top:5px;padding: 0px;">
        <table style="width: calc(100% );">
        <tr style="width: 100%;text-align: center;">
            <td style="font-size:1.25em;font-family:monospace;display: inline-block;text-align: center;width:35%;padding: 0px;border-bottom:0px">
            <img style="float: left; width: 100%;padding-bottom: 10px;" alt="paper thumbnail" src="./quasimetric_rl/assets/images/quasimetric_structure.png">
            Quasimetric Geometry
            </td><td style="font-size:2.9em;font-family:monospace;display: inline-block;text-align: center;width:5%;padding: 0px;border-bottom:0px">
            +
            </td><td style="font-size:1.25em;font-family:monospace;display: inline-block;text-align: center;vertical-align: bottom;width:33%;padding: 0px;border-bottom:0px;"><div style="padding-bottom: 3px" >
            <video src="https://user-images.githubusercontent.com/5674597/229619483-4e565dee-7b69-45a6-8f81-f21647f0df71.mp4" controls="controls" autoplay="true" loop="true" muted="muted" class="d-block rounded-bottom-2 border-top width-fit" style="width:100%">  </video></div>
            A Novel Objective<br>
            <div style="font-size: 0.67em;padding:0px;padding-top: 0px;">(Push apart <span style="color:rgb(217, 0, 0);font-weight: bold;">start state</span> and <span style="color:rgb(217, 0, 0);font-weight: bold;">goal</span><br>while maintaining local distances)</div>
            </td><td style="font-size:2.9em;font-family:monospace;display: inline-block;text-align: center;width:5%;padding: 0px;border-bottom:0px">
            =
            </td><td style="font-size:1.2em;font-family:monospace;display: inline-block;text-align: center;width:22%;padding: 0px;border-bottom:0px">
            Optimal Value $V^*$<br><span style="color:#97999c; font-weight: 200;font-style: italic;">AND</span><br>High-Performing<br>Goal-Reaching Agents
            </td>
        </tr>
        </table>
    </div>

2. **Improved Representation of Asymmetrical Distances with Interval Quasimetric Embeddings**<br />[[<span class="small__tt">**NeurIPS 2022 NeurReps Workshop**{: .colorful}</span>](https://www.neurreps.org/)] [[Project Page](./interval_quasimetric_embedding){: .small__tt}] [[arXiv](https://arxiv.org/abs/2211.15120){: .small__tt}] [[PyTorch Package for Quasimetric Learning](https://github.com/quasimetric-learning/torch-quasimetric){: .small__tt}] <br />
    **Tongzhou Wang**, Phillip Isola
    {: style="margin-bottom: 0"}

    <img src="./interval_quasimetric_embedding/images/iqe_compute_nobg.png" alt="computing-iqe" style="width:97%;margin-top:2.5%">

3. **Denoised MDPs: Learning World Models Better Than The World**<br />[<span class="small__tt">**ICML 2022**{: .colorful}</span>] [[Project Page](./denoised_mdp){: .small__tt}] [[arXiv](https://arxiv.org/abs/2206.15477){: .small__tt}] [[code](https://github.com/facebookresearch/denoised_mdp){: .small__tt}] <br />
    **Tongzhou Wang**, Simon S. Du, Antonio Torralba, Phillip Isola, Amy Zhang, Yuandong Tian
    {: style="margin-bottom: 0"}

    <video src="https://user-images.githubusercontent.com/5674597/173155667-d4bcc7af-1f12-4ba3-a733-ef9d5f631c96.mp4" controls="controls" autoplay="true" loop="true" muted="muted" class="d-block rounded-bottom-2 border-top width-fit" style="width:100%">  </video>

4. **On the Learning and Learnability of Quasimetrics**<br />[<span class="small__tt">**ICLR 2022**{: .colorful}</span>] [[Project Page](/quasimetric){: .small__tt}] [[arXiv](https://arxiv.org/abs/2206.15478){: .small__tt}] [[OpenReview](https://openreview.net/forum?id=y0VvIg25yk){: .small__tt}] [[code](https://github.com/SsnL/poisson_quasimetric_embedding){: .small__tt}] <br />
    **Tongzhou Wang**, Phillip Isola
    {: style="margin-bottom: 0"}

    <img src="./quasimetric/images/function_spaces_cropped.png" alt="quasimetric-function-spaces" style="width:100%">

5. **Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere**<br />[<span class="small__tt">**ICML 2020**{: .colorful}</span>] [[Project Page](/hypersphere){: .small__tt}] [[arXiv](https://arxiv.org/abs/2005.10242){: .small__tt}] [[code](https://github.com/SsnL/align_uniform){: .small__tt}] <br />
    **Tongzhou Wang**, Phillip Isola

    <div style="display: flex; width: 100%;margin-top: -0.5em">
    <div style="width:100%; height: 260px; display:flex">
        <div style="width:310px;border-bottom: 0px;padding:0px;vertical-align: bottom;text-align: left">
            <img style="max-width:100%;max-height:100%;" src="/assets/images/hypersphere_stl10_scatter_linear_output.png" alt="hypersphere_stl10_scatter_linear_output" />
        </div>
        <div style="width:3%"></div>
        <div style="width:calc(100% - 3% - 340px);border-bottom: 0px;padding-left:0px;padding-right:0px;text-align:right;vertical-align: bottom;display:contents;font-size:0.78em">
            <div style="font-size: 0.735em;display: inline-block;text-align:left;padding-bottom:1.5em;max-width:100%;max-height:100%;align-self:flex-end" >
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
        </div>
      </div>
    </div>

1. **Dataset Distillation**<br />[[Project Page](/dataset_distillation){: .small__tt}] [[arXiv](https://arxiv.org/abs/1811.10959){: .small__tt}] [[code](https://github.com/SsnL/dataset-distillation){: .small__tt}]  [[DD Papers](https://github.com/Guang000/Awesome-Dataset-Distillation){: .small__tt}] <br />
    **Tongzhou Wang**, Jun-Yan Zhu, Antonio Torralba, Alexei A. Efros

    ![dataset_distillation_fixed_mnist](/assets/images/dataset_distillation_fixed_mnist.png){: style="width:100%;"}

2. **Meta-Learning MCMC Proposals**<br />[<span class="small__tt">**NeurIPS 2018**{: .colorful}</span>] [<span class="small__tt">**PROBPROG 2018**{: .colorful}</span>] [[ICML 2017 AutoML Workshop Oral](./automl_17/slides.pdf){: .small__tt}] [[arXiv](https://arxiv.org/abs/1708.06040){: .small__tt}] <br />
    **Tongzhou Wang**, Yi Wu, David A. Moore, Stuart J. Russell

    ![meta_learning_mcmc_gmm_trace](/assets/images/meta_learning_mcmc_gmm_trace.png){: style="width:100%;"}

3. **Learning to Synthesize a 4D RGBD Light Field from a Single Image**<br />[<span class="small__tt">**ICCV 2017**{: .colorful}</span>] [[arXiv](https://arxiv.org/abs/1708.03292){: .small__tt}] <br />
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