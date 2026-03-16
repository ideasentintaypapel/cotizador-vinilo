import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import io
import math
import plotly.graph_objects as go

# --- CONFIGURACIÓN GLOBAL ---
st.set_page_config(page_title="Cotizador de Vinilo Pro", layout="wide")

# --- FUNCIONES AUXILIARES ---

def calcular_escala_inteligente(img_array, ancho_cm, max_megapixels=2.5):
    """
    Analiza la complejidad de la imagen y calcula un tamaño seguro 
    para evitar errores de Out-Of-Memory en la nube.
    """
    h_orig, w_orig = img_array.shape[:2]
    # -- PRE-ESCALADO DE SEGURIDAD (OOM prevention) --
    mp_orig = (w_orig * h_orig) / 1_000_000
    if mp_orig > max_megapixels:
        factor_seguridad = math.sqrt(max_megapixels / mp_orig)
        w_safe = int(w_orig * factor_seguridad)
        h_safe = int(h_orig * factor_seguridad)
        img_analisis = cv2.resize(img_array, (w_safe, h_safe), interpolation=cv2.INTER_AREA)
    else:
        img_analisis = img_array

    # 1. Análisis rápido de detalle (Varianza del Laplaciano) - Sobre imagen segura
    gris = cv2.cvtColor(img_analisis, cv2.COLOR_RGB2GRAY)
    varianza_bordes = cv2.Laplacian(gris, cv2.CV_64F).var()

    # 2. Asignación dinámica de PPI según complejidad
    if varianza_bordes < 300: 
        ppi_ideal = 75.0   # Trazos muy simples, logos planos
    elif varianza_bordes < 1000:
        ppi_ideal = 150.0  # Complejidad media
    else:
        ppi_ideal = 300.0  # Alto detalle, texturas complejas

    # 3. Cálculo de dimensiones target ideal
    ancho_px_target = int((ancho_cm / 2.54) * ppi_ideal)
    
    # 3.5 PREVENIR UPSCALING (El origen del problema de memoria)
    # Agrandar artificialmente una imagen digital no da más detalles físicos,
    # solo multiplica exponencialmente la RAM necesaria y crashea el servidor.
    if ancho_px_target > w_orig:
        ancho_px_target = w_orig

    escala = ancho_px_target / w_orig if w_orig > 0 else 1.0
    h_target = int(h_orig * escala)

    # 4. Freno de emergencia (OOM prevention vía Max Megapixels)
    mp_target = (ancho_px_target * h_target) / 1_000_000
    if mp_target > max_megapixels:
        factor_reduccion = math.sqrt(max_megapixels / mp_target)
        ancho_px_target = int(ancho_px_target * factor_reduccion)
        h_target = int(h_target * factor_reduccion)

    # 5. Calculamos el PPI que REALMENTE va a tener la imagen procesada
    ppi_aplicado = (ancho_px_target / ancho_cm) * 2.54 if ancho_cm > 0 else 75.0

    # Asegurar dimensiones válidas para evitar crash en cv2.resize
    ancho_px_target = max(1, ancho_px_target)
    h_target = max(1, h_target)

    return ancho_px_target, h_target, ppi_aplicado


def generar_pdf(img_base, df_activos, c_mat, c_dep, c_limp, c_vert, c_finas, total_final, cant_A, cant_M, cant_P, cant_F):
    img_export = img_base.copy()
    for idx, row in df_activos.iterrows():
        cat = row['Cat']
        if cat == 'A': color = (0, 0, 255)       
        elif cat == 'M': color = (255, 0, 255)   
        elif cat == 'P': color = (255, 0, 0)     
        elif cat == 'F': color = (0, 255, 255)   
        else: color = (255, 255, 255)
        
        cv2.circle(img_export, (int(row['cx']), int(row['cy'])), 14, color, -1)
        cv2.putText(img_export, row['Report_ID'], (int(row['cx']) - 10, int(row['cy']) + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    pil_img = Image.fromarray(img_export)
    img_buffer = io.BytesIO()
    pil_img.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    img_reader = ImageReader(img_buffer)
    
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=A4)
    ancho_a4, alto_a4 = A4
    
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, alto_a4 - 50, "Reporte Técnico de Auditoría - Corte Vinilo")
    
    img_w, img_h = pil_img.size
    aspect = img_h / float(img_w)
    print_w = ancho_a4 - 100
    print_h = print_w * aspect
    if print_h > (alto_a4 / 2):
        print_h = alto_a4 / 2
        print_w = print_h / aspect
        
    c.drawImage(img_reader, 50, alto_a4 - 80 - print_h, width=print_w, height=print_h)
    
    y_text = alto_a4 - 110 - print_h
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_text, "Desglose de Costos Operativos:")
    c.setFont("Helvetica", 10)
    y_text -= 20
    c.drawString(50, y_text, f"Material Base: ${c_mat:.2f}")
    c.drawString(50, y_text - 15, f"Depilados Estándar ({cant_A}x): ${c_dep:.2f}")
    c.drawString(50, y_text - 30, f"Limpieza Microperforaciones ({cant_M}x): ${c_limp:.2f}")
    c.drawString(50, y_text - 45, f"Recargo Riesgo Vértices ({cant_P}x): ${c_vert:.2f}")
    c.drawString(50, y_text - 60, f"Procesamiento Áreas Finas ({cant_F}x): ${c_finas:.2f}")
    
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_text - 95, f"PRESUPUESTO FINAL RECOMENDADO: ${total_final:.2f}")
    
    c.save()
    pdf_buffer.seek(0)
    return pdf_buffer


# --- FASE 1: SIDEBAR (CONFIGURACIÓN ESTÁTICA) ---
st.sidebar.title("⚙️ Configuración")

with st.sidebar.expander("1. Carga y Medidas", expanded=True):
    uploaded_file = st.file_uploader("Diseño (JPG, PNG, SVG)", type=["jpg", "jpeg", "png", "svg"])
    ancho_cm = st.number_input("Ancho real (cm)", min_value=1.0, value=15.0, step=0.5)

with st.sidebar.expander("2. Digitalización y Offset", expanded=False):
    num_colores = st.slider("Colores a extraer", 1, 5, 2)
    corregir_luz = st.checkbox("Corregir sombras/iluminación", value=True)
    suavizado_bordes = st.slider("Suavizado de bordes (Blur)", 3, 21, 5, step=2)
    ignorar_fondo = st.checkbox("Ignorar color de fondo (esquinas)", value=True)

with st.sidebar.expander("3. Parámetros de Auditoría", expanded=False):
    umbral_micro = st.slider("Umbral Microagujeros (Diámetro mm)", 0.1, 5.0, 2.0, 0.1)
    umbral_angulo = st.slider("Ángulo crítico de vértice (°)", 5, 45, 15)
    st.markdown("---")
    grosor_min_mm = st.slider("Grosor Áreas Finas (< mm)", 0.1, 5.0, 1.0, 0.1)
    longitud_min_mm = st.slider("Longitud Mín. Área Fina (> mm)", 1.0, 20.0, 5.0, 0.5)

with st.sidebar.expander("4. Costos", expanded=False):
    costo_m2 = st.number_input("Costo Vinilo ($ por m2)", value=9000.0)
    costo_agujero = st.number_input("Costo por Depilado ($)", value=50.0)
    costo_limpieza = st.number_input("Costo Limpieza Micro ($)", value=20.0)
    recargo_angulo = st.number_input("Recargo por Vértice Crítico ($)", value=10.0)
    costo_finas = st.number_input("Costo Proc. Área Fina ($)", value=10.0)
    margen_ganancia = st.slider("Margen de Ganancia (%)", 50, 500, 200)

ejecutar = st.sidebar.button("🚀 Ejecutar/Actualizar Digitalización", type="primary", use_container_width=True)

# --- INICIALIZACIÓN DE ESTADOS ---
if "procesado" not in st.session_state:
    st.session_state.procesado = False
    st.session_state.df_riesgos = pd.DataFrame()
    st.session_state.img_base = None
    st.session_state.img_contornos = None
    st.session_state.area_m2 = 0.0

# --- PROCESAMIENTO PESADO ---
if ejecutar and uploaded_file is not None:
    with st.spinner("Procesando imagen, auditando y detectando áreas finas..."):
        if uploaded_file.name.lower().endswith('.svg'):
            drawing = svg2rlg(uploaded_file)
            img_buffer = io.BytesIO()
            # Reducimos drásticamente los DPI (de 300 a 72) porque los SVG de vinilos (metros de largo) 
            # a 300 DPI generan imágenes en RAM de varios Gigabytes y crashean el servidor (OOM).
            renderPM.drawToFile(drawing, img_buffer, fmt="PNG", dpi=72)
            img_buffer.seek(0)
            image = Image.open(img_buffer).convert("RGB")
        else:
            image = Image.open(uploaded_file).convert("RGB")
        
        img_array = np.array(image)

        # --- NUEVA LÓGICA DE ESCALADO INTELIGENTE ---
        ancho_px_target, alto_px_target, ppi_actual = calcular_escala_inteligente(img_array, ancho_cm)

        # Escalamos para evitar OOM si la original es muy grande o la diferencia es significativa
        img_mp = (img_array.shape[1] * img_array.shape[0]) / 1_000_000
        if abs(ancho_px_target - img_array.shape[1]) > 50 or img_mp > 2.5:
            # INTER_AREA reduce mejor, sin generar ruido artificial y con menos RAM que CUBIC
            img_procesada = cv2.resize(img_array, (ancho_px_target, alto_px_target), interpolation=cv2.INTER_AREA)
        else:
            img_procesada = img_array.copy()
            
        # Actualizamos el factor de escala por si lo necesitas más adelante
        factor_escala = img_procesada.shape[1] / img_array.shape[1]
        # ----------------------------------------------        
        if corregir_luz:
            lab = cv2.cvtColor(img_procesada, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(l)
            img_procesada = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)

        Z = np.float32(img_procesada.reshape((-1, 3)))
        # Reducimos attempts de 10 a 3 para evitar OOM (Out-of-memory) con KMeans en Streamlit
        _, label, center = cv2.kmeans(Z, num_colores, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 3, cv2.KMEANS_RANDOM_CENTERS)
        img_cuantizada = np.uint8(center)[label.flatten()].reshape((img_procesada.shape))
        bg_color = img_cuantizada[0, 0] if ignorar_fondo else None
        
        perimetro_critico_px = ((umbral_micro * math.pi) / 25.4) * ppi_actual
        grosor_px = int((grosor_min_mm / 25.4) * ppi_actual)
        grosor_px = max(3, grosor_px | 1) 
        longitud_px = (longitud_min_mm / 25.4) * ppi_actual
        kernel_finas = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grosor_px, grosor_px))
        
        lista_riesgos = []
        conteo = {'A': 1, 'M': 1, 'P': 1, 'F': 1}
        img_contornos_verdes = np.zeros_like(img_procesada, dtype=np.uint8)

        for _color in center:
            color_uint8 = np.uint8(_color)
            if ignorar_fondo and np.array_equal(color_uint8, bg_color): continue
            
            mask = cv2.inRange(img_cuantizada, np.array(color_uint8), np.array(color_uint8))
            mask_blur = cv2.GaussianBlur(mask, (suavizado_bordes, suavizado_bordes), 0)
            contours, hierarchy = cv2.findContours(mask_blur, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            
            # Áreas Finas
            _, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            opened = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel_finas)
            thin_mask = cv2.subtract(bin_mask, opened)
            contours_thin, _ = cv2.findContours(thin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt_f in contours_thin:
                if cv2.arcLength(cnt_f, True) > longitud_px:
                    M = cv2.moments(cnt_f)
                    cx, cy = (int(M['m10']/M['m00']), int(M['m01']/M['m00'])) if M['m00'] != 0 else (cnt_f[0][0][0], cnt_f[0][0][1])
                    lista_riesgos.append({"Audit_ID": f"F{conteo['F']}", "Tipo": "Área Fina", "Incluir": True, "cx": cx, "cy": cy, "Cat": "F"})
                    conteo['F'] += 1

            # Resto de Riesgos
            if hierarchy is not None:
                for i, cnt in enumerate(contours):
                    cv2.drawContours(img_contornos_verdes, [cnt], -1, (0, 255, 0), 2)
                    es_agujero = hierarchy[0][i][3] != -1
                    arc_len = cv2.arcLength(cnt, True)
                    M = cv2.moments(cnt)
                    cx, cy = (int(M['m10']/M['m00']), int(M['m01']/M['m00'])) if M['m00'] != 0 else (cnt[0][0][0], cnt[0][0][1])
                    
                    if es_agujero:
                        if arc_len <= perimetro_critico_px:
                            lista_riesgos.append({"Audit_ID": f"M{conteo['M']}", "Tipo": "Microperforación", "Incluir": True, "cx": cx, "cy": cy, "Cat": "M"})
                            conteo['M'] += 1
                        else:
                            lista_riesgos.append({"Audit_ID": f"A{conteo['A']}", "Tipo": "Agujero Depilado", "Incluir": True, "cx": cx, "cy": cy, "Cat": "A"})
                            conteo['A'] += 1
                    else:
                        approx = cv2.approxPolyDP(cnt, 0.01 * arc_len, True)
                        for j in range(len(approx)):
                            p1, p2, p3 = approx[j-1][0], approx[j][0], approx[(j+1)%len(approx)][0]
                            ba, bc = p1-p2, p3-p2
                            norm_ba, norm_bc = np.linalg.norm(ba), np.linalg.norm(bc)
                            if norm_ba > 0 and norm_bc > 0:
                                cosine_angle = np.clip(np.dot(ba,bc)/(norm_ba*norm_bc), -1.0, 1.0)
                                if np.degrees(np.arccos(cosine_angle)) < umbral_angulo:
                                    lista_riesgos.append({"Audit_ID": f"P{conteo['P']}", "Tipo": "Vértice Crítico", "Incluir": True, "cx": p2[0], "cy": p2[1], "Cat": "P"})
                                    conteo['P'] += 1

        st.session_state.df_riesgos = pd.DataFrame(lista_riesgos) if lista_riesgos else pd.DataFrame(columns=["Audit_ID", "Tipo", "Incluir", "cx", "cy", "Cat"])
        st.session_state.img_base = img_procesada
        st.session_state.img_contornos = img_contornos_verdes
        st.session_state.procesado = True
        st.session_state.fuerza_render = True
        st.session_state.area_m2 = ((img_procesada.shape[1] / ppi_actual * 2.54) * (img_procesada.shape[0] / ppi_actual * 2.54)) / 10000

# --- FASE 2: PANEL DE AUDITORÍA (NUEVA DISTRIBUCIÓN) ---
if st.session_state.procesado:
    st.title("✂️ Panel de Auditoría Profesional")
    
    # 1. Definición de layout principal
    col_control, col_visor = st.columns([1, 2.5])
    
    with col_control:
        st.subheader("🛠️ Puntos de Auditoría")
        df_edit = st.data_editor(
            st.session_state.df_riesgos[["Audit_ID", "Tipo", "Incluir"]],
            column_config={"Incluir": st.column_config.CheckboxColumn("Cobrar/Ejecutar", default=True)},
            disabled=["Audit_ID", "Tipo"],
            hide_index=True,
            width='stretch'
        )
        
        # Cálculos en tiempo real
        df_activos = df_edit[df_edit["Incluir"] == True].copy()
        if not df_activos.empty:
            df_activos = df_activos.merge(st.session_state.df_riesgos[['Audit_ID', 'cx', 'cy', 'Cat']], on='Audit_ID')
            for cat in ['A', 'M', 'P', 'F']:
                mask = df_activos['Cat'] == cat
                if mask.any():
                    df_activos.loc[mask, 'Report_ID'] = [f"{cat}{i+1}" for i in range(mask.sum())]
            
            cant_A = (df_activos['Cat'] == 'A').sum()
            cant_M = (df_activos['Cat'] == 'M').sum()
            cant_P = (df_activos['Cat'] == 'P').sum()
            cant_F = (df_activos['Cat'] == 'F').sum()
        else:
            cant_A = cant_M = cant_P = cant_F = 0
            df_activos['Report_ID'] = []
            
        c_mat = st.session_state.area_m2 * costo_m2
        c_dep = cant_A * costo_agujero
        c_limp = cant_M * costo_limpieza
        c_vert = cant_P * recargo_angulo
        c_finas = cant_F * costo_finas
        total_final = (c_mat + c_dep + c_limp + c_vert + c_finas) * (1 + (margen_ganancia/100))

    # 2. Inyección Dinámica en la Barra Lateral
    with st.sidebar:
        st.markdown("---")
        with st.expander("💰 5. Presupuesto Final", expanded=True):
            st.write(f"**Material:** ${c_mat:.2f}")
            st.write(f"**Depilados ({cant_A}):** ${c_dep:.2f}")
            st.write(f"**Limpieza Micro ({cant_M}):** ${c_limp:.2f}")
            st.write(f"**Vértices ({cant_P}):** ${c_vert:.2f}")
            st.write(f"**Áreas Finas ({cant_F}):** ${c_finas:.2f}")
            st.metric("TOTAL A COBRAR", f"${total_final:.2f}")

        with st.expander("👁️ 6. Gestión de Capas", expanded=True):
            render_sincronizado = st.checkbox("⚡ Renderizado Sincronizado (Auto)", value=True, help="Actívalo para ver los cambios en vivo. Desactívalo para evitar parpadeos y actualizar manualmente.")
            st.markdown("---")
            mostrar_lineas = st.checkbox("🟢 Líneas de Corte", value=True)
            mostrar_agujeros = st.checkbox("🔴 Agujeros (A)", value=True)
            mostrar_micro = st.checkbox("🟣 Microperforaciones (M)", value=True)
            mostrar_vertices = st.checkbox("🔵 Vértices (P)", value=True)
            mostrar_finas = st.checkbox("🟡 Áreas Finas (F)", value=True)
            st.markdown("---")
            vista = st.radio("Modo IDs:", ["Ver todos", "Ver activos", "Ocultar IDs"])

    # 3. Renderizado de Botones Inferiores
    st.markdown("---")
    c_btn1, c_btn2, _ = st.columns([1, 1, 1.5]) 
    
    # Solo mostramos el botón de actualizar si NO está sincronizado
    btn_actualizar = False
    if not render_sincronizado:
        btn_actualizar = c_btn1.button("🔄 Actualizar Renderizado", type="primary", use_container_width=True)
    
    if not df_activos.empty:
        pdf_bytes = generar_pdf(st.session_state.img_base, df_activos, c_mat, c_dep, c_limp, c_vert, c_finas, total_final, cant_A, cant_M, cant_P, cant_F)
        c_btn2.download_button(label="📄 Exportar Reporte Técnico", data=pdf_bytes, file_name="Reporte_Vinilo.pdf", mime="application/pdf", use_container_width=True)

    # 4. Renderizado del Visor
    with col_visor:
        st.subheader("👁️ Visor Unificado")
        
        # Lógica inteligente para saber cuándo reconstruir la imagen
        debe_renderizar = False
        if "fig" not in st.session_state:
            debe_renderizar = True
        elif st.session_state.get("fuerza_render", False):
            debe_renderizar = True
        elif render_sincronizado:
            debe_renderizar = True
        elif btn_actualizar:
            debe_renderizar = True

        if debe_renderizar:
            if "fuerza_render" in st.session_state: st.session_state.fuerza_render = False

            img_visual = cv2.addWeighted(st.session_state.img_base, 0.7, st.session_state.img_contornos, 0.8, 0) if mostrar_lineas else st.session_state.img_base
            fig = go.Figure()
            fig.add_trace(go.Image(z=img_visual))
            
            df_plot = st.session_state.df_riesgos.copy()
            if vista == "Ver activos":
                df_plot = df_plot[df_plot["Audit_ID"].isin(df_activos["Audit_ID"])]
            elif vista == "Ocultar IDs":
                df_plot = pd.DataFrame()
                
            if not df_plot.empty:
                if mostrar_agujeros:
                    df_A = df_plot[df_plot['Cat'] == 'A']
                    if not df_A.empty: fig.add_trace(go.Scatter(x=df_A['cx'], y=df_A['cy'], mode='markers+text', marker=dict(size=14, color='rgba(255,0,0,0.6)', line=dict(width=2, color='white')), text=df_A['Audit_ID'], textposition="top center", textfont=dict(color='white', size=14)))
                if mostrar_micro:
                    df_M = df_plot[df_plot['Cat'] == 'M']
                    if not df_M.empty: fig.add_trace(go.Scatter(x=df_M['cx'], y=df_M['cy'], mode='markers+text', marker=dict(size=10, color='rgba(255,0,255,0.6)', line=dict(width=1, color='white')), text=df_M['Audit_ID'], textposition="top center", textfont=dict(color='white', size=13)))
                if mostrar_vertices:
                    df_P = df_plot[df_plot['Cat'] == 'P']
                    if not df_P.empty: fig.add_trace(go.Scatter(x=df_P['cx'], y=df_P['cy'], mode='markers+text', marker=dict(size=8, color='rgba(0,191,255,0.8)', line=dict(width=1, color='black')), text=df_P['Audit_ID'], textposition="top center", textfont=dict(color='white', size=12), hoverinfo="text"))
                if mostrar_finas:
                    df_F = df_plot[df_plot['Cat'] == 'F']
                    if not df_F.empty: fig.add_trace(go.Scatter(x=df_F['cx'], y=df_F['cy'], mode='markers+text', marker=dict(size=10, color='rgba(255,255,0,0.8)', line=dict(width=1, color='black')), text=df_F['Audit_ID'], textposition="top center", textfont=dict(color='white', size=12), hoverinfo="text"))

            fig.update_layout(height=700, margin=dict(l=0,r=0,t=0,b=0), uirevision="constant", showlegend=False)
            fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
            st.session_state.fig = fig

        st.plotly_chart(st.session_state.fig, width='stretch', config={'scrollZoom': True})

else:
    if uploaded_file is None:
        st.info("👈 Sube un diseño y ajusta la configuración para comenzar.")