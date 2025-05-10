import streamlit as st
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import io

st.set_page_config(page_title="Live K-Means Image Compressor", layout="centered")
st.title("🎨 Real-Time Image Compression with K-Means")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image) / 255.0
    original_shape = image_np.shape
    pixels = image_np.reshape(-1, 3)

    # Compute max number of clusters (colors)
    K_max = min(256, len(np.unique(pixels, axis=0)))

    compressibility = st.slider("📉 Select compressibility (%)", 0, 100, 50)
    K = max(2, int(K_max * (1 - compressibility / 100)))

    st.markdown(f"🧠 **Using K = {K} colors**")

    # Compress image using K-Means
    kmeans = KMeans(n_clusters=K, random_state=42, n_init='auto')
    kmeans.fit(pixels)
    palette = kmeans.cluster_centers_
    compressed_pixels = palette[kmeans.labels_].reshape(original_shape)

    # Compression statistics
    num_pixels = pixels.shape[0]
    original_bits = num_pixels * 24
    centroid_bits = K * 24
    index_bits = int(np.ceil(np.log2(K)))
    compressed_bits = centroid_bits + num_pixels * index_bits

    original_kb = original_bits / (8 * 1024)
    compressed_kb = compressed_bits / (8 * 1024)
    ratio = compressed_kb / original_kb

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🖼️ Original Image")
        st.image(image, use_column_width=True)
    with col2:
        st.subheader("🖼️ Compressed Image")
        st.image((compressed_pixels * 255).astype(np.uint8), use_column_width=True)

    st.subheader("📊 Compression Stats")
    st.markdown(f"- **Original Size:** {original_kb:.2f} KB")
    st.markdown(f"- **Compressed Size:** {compressed_kb:.2f} KB")
    st.markdown(f"- **Compression Ratio:** {ratio:.2f}")

    # Download button
    compressed_img = Image.fromarray((compressed_pixels * 255).astype(np.uint8))
    buf = io.BytesIO()
    compressed_img.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button("📥 Download Compressed Image", data=byte_im,
                       file_name="compressed_image.png", mime="image/png")
