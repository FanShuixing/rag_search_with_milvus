# 1. 选择基础镜像
# 使用 Python 3.10 的官方精简版，体积更小
FROM python:3.10-slim

# 2. 设置工作目录
WORKDIR /app

# 3. 优化点：先单独复制依赖文件
# 这样当 requirements.txt 没变时，Docker 会复用缓存，不会重新下载包
COPY requirements.txt .

# 4. 安装 Python 依赖
# --no-cache-dir 参数用于减小镜像体积
RUN pip install --no-cache-dir -r requirements.txt

# 5. 最后复制项目所有代码
# 因为代码变动最频繁，所以放在最后
COPY . .

# 6. 暴露 Streamlit 默认端口
EXPOSE 8501

# 7. 定义容器启动命令

CMD ["streamlit", "run", "app.py"]