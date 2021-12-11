FROM nvidia/cuda:10.2-devel-ubuntu18.04

# Configure build tools
ENV CMAKE_MAJOR_VERSION=3.20
ENV CMAKE_VERSION=3.20.0

# Configure package versions of TensorFlow, PyTorch, CUDA, cuDNN and NCCL
ENV TENSORFLOW_VERSION=2.3.0
ENV PYTORCH_VERSION=1.8.1+cu102
ENV TORCHVISION_VERSION=0.9.1+cu102
ENV TORCHAUDIO_VERSION=0.8.1
ENV CUDA_VERSION=10.2
ENV CUDNN_MAJOR_VERSION=8
ENV CUDNN_VERSION=8.2.1.32-1+cuda10.2
ENV NCLL_MAJOR_VERSION=2
ENV NCCL_VERSION=2.9.9-1+cuda10.2
ENV DGL_VERSION=cu102

# Configure Python version 2.7 or 3.6
ARG python=3.6
ENV PYTHON_VERSION=${python}

# Configure OpenMPI version
ENV OPENMPI_MAJOR_VERSION=4.1
ENV OPENMPI_VERSION=4.1.1

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

# Get rid of the debconf messages
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# Update repository
 RUN apt-get update

# Configure timezone
RUN ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime
RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata

# Install and configure locales
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y locales
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8

# Install the essential packages
RUN apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        apt-utils \
        build-essential \
        g++-4.8 \
        cmake \
        git \
        curl \
        vim \
        wget \
        ccache \
        ca-certificates \
        libcudnn${CUDNN_MAJOR_VERSION}=${CUDNN_VERSION} \
        libcudnn${CUDNN_MAJOR_VERSION}-dev=${CUDNN_VERSION} \
        libnccl${NCLL_MAJOR_VERSION}=${NCCL_VERSION} \
        libnccl-dev=${NCCL_VERSION} \
        libjpeg-dev \
        libpng-dev \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        librdmacm1 \
        libibverbs1 \
        ibverbs-providers \
        libopenblas-dev \
        libopencv-dev

# Latest build tools
RUN cd /tmp && wget https://cmake.org/files/v${CMAKE_MAJOR_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz && tar -zxf cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz && mv cmake-${CMAKE_VERSION}-linux-x86_64 /usr/local/cmake

# Install Python
RUN if [[ "${PYTHON_VERSION}" == "3.6" ]]; then \
        apt-get install -y python${PYTHON_VERSION}-distutils; \
    fi
RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py && \
    pip install --upgrade pip wheel

# Install TensorFlow, Keras, PyTorch
RUN pip install future typing
RUN pip install --no-cache-dir numpy \
        tensorflow-gpu==${TENSORFLOW_VERSION} \
        keras \
        h5py
RUN pip install --no-cache-dir torch torchvision torchaudio -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
#RUN pip install torch===${PYTORCH_VERSION} torchvision===${TORCHVISION_VERSION} torchaudio===${TORCHAUDIO_VERSION} -f https://download.pytorch.org/whl/torch_stable.html

# Install apex
RUN git clone https://github.com/NVIDIA/apex /usr/local/apex && \
    cd /usr/local/apex && \
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# Install Graph-Learning packages
RUN pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-${PYTORCH_VERSION}.html && pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-${PYTORCH_VERSION}.html && pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-${PYTORCH_VERSION}.html && pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-${PYTORCH_VERSION}.html && pip install torch-geometric
RUN pip install dgl-${DGL_VERSION}

# Install Graph-Learning models
RUN pip install torchbiggraph
RUN pip install ampligraph
RUN pip install karateclub
RUN pip install rdflib SPARQLWrapper

# Install Faiss
RUN pip install numpy
RUN apt-get install -y swig
RUN git clone https://github.com/facebookresearch/faiss.git /tmp/faiss && \
    cd /tmp/faiss && \
    /usr/local/cmake/bin/cmake -B build . -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=ON -DCUDAToolkit_ROOT=/usr/local/cuda && make -C build -j $(nproc) faiss && make -C build -j $(nproc) swigfaiss && \
    cd build/faiss/python && python setup.py install && \
    rm -rf /tmp/faiss

# Install Open MPI
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://www.open-mpi.org/software/ompi/v${OPENMPI_MAJOR_VERSION}/downloads/openmpi-${OPENMPI_VERSION}.tar.gz -q && \
    tar zxf openmpi-${OPENMPI_VERSION}.tar.gz && \
    cd openmpi-${OPENMPI_VERSION} && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

# Install Horovod
RUN ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
    HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_ALLOW_MIXED_GPU_IMPL=0 HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 \
    CFLAGS="-O2 -mavx -mfma" \
         pip install --no-cache-dir horovod && \
    ldconfig

# Install OpenSSH for MPI to communicate between containers
RUN apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

# Download source code
RUN mkdir /source && \
    cd /source && \
    git clone https://github.com/cskyan/biograph.git && \
    rm -rf biograph/.git
ENV PYTHONPATH=/source

# Prepare workspace
RUN mkdir /workspace
WORKDIR "/workspace"
