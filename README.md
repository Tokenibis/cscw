This repository contains source code for the analysis and document submitted to the 2024 CSWC (Computer-Supported Cooperative Work and Social Computing) conference call for poster presentations.

# Paper

Dependencies: texlive

`$ cd paper`

`$ pdflatex main.tex && bibtex main.aux && pdflatex main.tex && pdflatex main.tex`

The final document will be in `paper/main.pdf`

# Analysis

Dependencies: Python (>3.10)

`$ python -m spacy download en_core_web_lg`

`$ python analyze.py`

The inputs for the paper are: `org-shares.tex` and `cls-shares.tex`

# Sociogram

Available live at: `https://api.app.tokenibis.org/media/graphs/network.svg`
