FROM nvidia/cuda:10.1-cudnn7-devel

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
	python3-opencv ca-certificates python3-dev git wget sudo vim curl dumb-init locales net-tools openssl zsh neovim git git-lfs && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /data

ENV PATH="/data/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py && \
	rm get-pip.py

# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install torch torchvision tensorboard cython
RUN pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

RUN pip install 'git+https://github.com/facebookresearch/fvcore'
# install detectron2
# RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
ENV FORCE_CUDA="1"
# This will build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ENV TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
# RUN pip install -e detectron2_repo

# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"

##############################################################################################
## install VSCODE dependancies
COPY requirements.txt /tmp

WORKDIR /tmp
RUN pip install -r requirements.txt
RUN curl -O -L https://npmjs.org/install.sh

RUN git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ~/.zsh/zsh-syntax-highlighting
RUN echo "source ~/.zsh/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh" >> ~/.zshrc
RUN git clone https://github.com/zsh-users/zsh-autosuggestions ~/.zsh/zsh-autosuggestions
RUN echo "source ~/.zsh/zsh-autosuggestions/zsh-autosuggestions.zsh" >> ~/.zshrc
RUN curl -sLf https://spacevim.org/install.sh | bash
RUN curl -fsSL https://raw.github.com/robbyrussell/oh-my-zsh/master/tools/install.sh | bash
RUN sudo chsh -s `which zsh`

#VSCODE
WORKDIR /
RUN sudo wget https://github.com/cdr/code-server/releases/download/2.1665-vsc1.39.2/code-server2.1665-vsc1.39.2-linux-x86_64.tar.gz
RUN sudo tar -xvf code-server2.1665-vsc1.39.2-linux-x86_64.tar.gz
RUN sudo mv code-server2.1665-vsc1.39.2-linux-x86_64 coder
RUN sudo chmod 777 -R coder
ENV PASSWORD "0000"

WORKDIR /data
RUN mkdir -p dataset
RUN mkdir -p ADD
RUN ln -s /data/dataset /data/ADD
RUN ln -s /data/dataset ~/ADD_dataset

#CleanUp
# RUN ldconfig && \
#     apt-get clean && \
#     apt-get autoremove && \
#     rm -rf /var/lib/apt/lists/* /tmp/* ~/*

WORKDIR /data/ADD