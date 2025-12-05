# Test-Time Consistency in Vision Language Models
[WACV 2026] Test-Time Consistency in Vision Language Models
[[Paper]()] [[HF]()] <br>
[Shih-Han Chou](https://shihhanchou.github.io/)*, [Shivam Chandhok](https://scholar.google.com/citations?user=ZER2BeIAAAAJ&hl=en)*, [James J. Little](https://www.cs.ubc.ca/~little/), [Leonid Sigal](https://www.cs.ubc.ca/~lsigal/)

To run test-time adaptation:
```bash
bash finetune_entropy_test_adaptive_steps.sh

# You can change the arguments in the bash file for different models and tasks.
# -n: the $n$th example in the datalist 
# -a: weight for loss
# -b: weight for loss
# -m: model name (choose from: llava, llava_next, qwen-vl, InternVL3, Idefics2)
# -t: task (choose from: rephrase, restyle, masking)
# -u: adaptive steps (default 4)
```

To do the evaluation:
```bash
# For rephrasing task (constant steps)
python metrics_rephrasing_constant.py -m $model_name$ -v $variants$
# For rephrasing task (adaptive steps)
python metrics_rephrasing_adaptive.py -m $model_name$ -v $variants$

# For masking task (constant steps)
python metrics_masking_constant.py -m $model_name$ -v $variants$
# For masking task (adaptive steps)
python metrics_masking_adaptive.py -m $model_name$ -v $variants$

# For restyling task (constant steps)
python metrics_style_constant.py -m $model_name$ -v $variants$
# For restyling task (adaptive steps)
python metrics_style_adaptive.py -m $model_name$ -v $variants$
```

## Acknowledgements

This work was funded, in part, by the Vector Institute for AI, Canada CIFAR AI Chairs, NSERC Canada Research Chair (CRC), and NSERC Discovery Grants. Resources used in preparing this research were provided, in part, by the Province of Ontario, the Government of Canada through CIFAR, the [Digital Research Alliance of Canada](https://vectorinstitute.ai/\#partners), companies sponsoring the Vector Institute, and Advanced Research Computing at the University of British Columbia. Additional hardware support was provided by John R. Evans Leaders Fund CFI grant and Compute Canada under the Resource Allocation Competition award.

## Citaton

If you find our work useful, please cite us using the following BibTeX entry.

```bibtex
@article{chou2025test,
  title={Test-Time Consistency in Vision Language Models},
  author={Chou, Shih-Han and Chandhok, Shivam and Little, James J and Sigal, Leonid},
  journal={arXiv preprint arXiv:2506.22395},
  year={2025}
}
```
