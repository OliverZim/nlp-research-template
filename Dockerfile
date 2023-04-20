# This Dockerfile produces a container with all dependencies installed into an environment called "research"
# Additionally, the container has a full-fledged micromamba installation, which is a faster drop-in replacement for conda
# When inside the container, you can install additional dependencies with `conda install <package>`, e.g. `conda install scipy`
# The actual installation is done by micromamba, we have simply provided an alias to link the conda command to micromamba

# Load micromamba container to copy from later
FROM --platform=linux/amd64 mambaorg/micromamba:1.3.1 as micromamba

# -----------------
# base image for amd64
# -----------------
FROM --platform=linux/amd64 nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as amd64
# Copy lockfile to container
COPY conda-lock.yml /locks/conda-lock.yml
# Install compiler for .compile() with PyTorch 2.0 and openssh-client & nano for devcontainers
RUN apt-get update && apt-get install -y gcc g++ openssh-client nano && apt-get clean all

# -----------------
# base image for ppc64le
# -----------------
FROM --platform=linux/ppc64le nvidia/cuda:11.8.0-cudnn8-runtime-ubi8 as ppc64le
# Copy ppc64le specififc lockfile to container
COPY ppc64le.conda-lock.yml /locks/conda-lock.yml
# Install compiler for .compile() with PyTorch 2.0
RUN yum install -y gcc gcc-c++ && yum clean all


# -----------------
# Final build image - we choose the correct base image based on the target architecture
# -----------------
ARG TARGETARCH
FROM $TARGETARCH as final
# From https://github.com/mamba-org/micromamba-docker#adding-micromamba-to-an-existing-docker-image
# The commands below add micromamba to an existing image to give the capability to ad-hoc install new dependencies

####################################################
######### Adding micromamba starts here ############
####################################################
USER root

# if your image defaults to a non-root user, then you may want to make the
# next 3 ARG commands match the values in your image. You can get the values
# by running: docker run --rm -it my/image id -a
ARG MAMBA_USER=mamba
ARG MAMBA_USER_ID=1000
ARG MAMBA_USER_GID=1000
ENV MAMBA_USER=$MAMBA_USER
ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV MAMBA_EXE="/bin/micromamba"

COPY --from=micromamba "$MAMBA_EXE" "$MAMBA_EXE"
COPY --from=micromamba /usr/local/bin/_activate_current_env.sh /usr/local/bin/_activate_current_env.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_shell.sh /usr/local/bin/_dockerfile_shell.sh
COPY --from=micromamba /usr/local/bin/_entrypoint.sh /usr/local/bin/_entrypoint.sh
COPY --from=micromamba /usr/local/bin/_activate_current_env.sh /usr/local/bin/_activate_current_env.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_initialize_user_accounts.sh /usr/local/bin/_dockerfile_initialize_user_accounts.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_setup_root_prefix.sh /usr/local/bin/_dockerfile_setup_root_prefix.sh

RUN /usr/local/bin/_dockerfile_initialize_user_accounts.sh && \
    /usr/local/bin/_dockerfile_setup_root_prefix.sh

USER $MAMBA_USER

SHELL ["/usr/local/bin/_dockerfile_shell.sh"]

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
# Optional: if you want to customize the ENTRYPOINT and have a conda
# environment activated, then do this:
# ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "my_entrypoint_program"]

# You can modify the CMD statement as needed....
CMD ["/bin/bash"]

####################################################
######### Adding micromamba stops here #############
####################################################

# Switch to root user to grant necessary permissions
USER root
# Give user permission to gcc
RUN chown $MAMBA_USER:$MAMBA_USER /usr/bin/gcc
# Necessary to prevent permission error when micromamba tries to install pip dependencies from lockfile
RUN chown $MAMBA_USER:$MAMBA_USER /locks/
RUN chown $MAMBA_USER:$MAMBA_USER /locks/conda-lock.yml
# Provide conda alias for micromamba
RUN echo "alias conda=micromamba" >> /usr/local/bin/_activate_current_env.sh

# Switch back to micromamba user
USER $MAMBA_USER
ARG TARGETPLATFORM
# Install dependencies from lockfile into environment, cache packages in /opt/conda/pkgs
RUN --mount=type=cache,target=${MAMBA_ROOT_PREFIX}/pkgs,id=conda-${TARGETPLATFORM} micromamba create --name research --yes --file /locks/conda-lock.yml
# Run this in a separate step to still keep the docker cache for future builds
RUN micromamba clean --all --yes

# Set conda-forge as default channel (otherwise no default channel is set)
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN micromamba config prepend channels conda-forge --env
# Disable micromamba banner at every command
RUN micromamba config set show_banner false --env
# Install optional tricky pip dependencies that do not work with conda-lock
# RUN micromamba run -n research pip install example-dependency --no-deps --no-cache-dir

# Use our environment as default
ENV ENV_NAME=research
