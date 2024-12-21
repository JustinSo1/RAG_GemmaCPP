<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->


<!-- PROJECT LOGO -->
<br />
<div align="center">

<h3 align="center">RAG_GemmaCPP</h3>

  <p align="center">
    Naive RAG Pipeline using gemma.cpp and scann
    <br />
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->

## About The Project

Naive RAG pipeline with Python

- LLM: [gemma.cpp](https://github.com/google/gemma.cpp)
- Vector
  Search: [ScaNN (Scalable Nearest Neighbors)](https://github.com/google-research/google-research/tree/master/scann)
- Dataset: [Mini Wikipedia](https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia)

### Built With

* Python 3.1x
* [ScaNN (Scalable Nearest Neighbors)](https://github.com/google-research/google-research/tree/master/scann)
* HuggingFace
* [gemma.cpp](https://github.com/google/gemma.cpp)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->

## Getting Started

```sh
python main.py
```

### Prerequisites

- Python 3.1x
- Able to pip install scann
- Gemma 1 weights and tokenizer gotten from https://www.kaggle.com/models/google/gemma/gemmaCpp

### Installation

1. Install packages
   ```sh
   pip install -r requirements.txt
   ```

2. Install pygemma from source
   ```sh
    cd gemma-cpp-python
    pip install .
   ```


<p align="right">(<a href="#readme-top">back to top</a>)</p>