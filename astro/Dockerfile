# Use the Astronomer base image
FROM quay.io/astronomer/astro-runtime:12.1.1

# Install libgomp1 for LightGBM
USER root
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Switch back to the astro user
USER astro
