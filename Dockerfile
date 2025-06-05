FROM node:20-alpine  

WORKDIR /app

# Copy package files first (optimizes Docker layer caching)
COPY package.json package-lock.json ./

# Install dependencies including ts-node
RUN npm install && \
    npm install -g ts-node typescript  

# Copy remaining files (excluding node_modules via .dockerignore)
COPY . .

# Verify ts-node is available
RUN ts-node --version

# Run your application
CMD ["ts-node", "/app/index.ts"] 