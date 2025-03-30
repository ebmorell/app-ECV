
import streamlit as st
import joblib
import numpy as np

# Cargar modelo
modelo = joblib.load("/content/modelo_ECV_RF.pkl")

st.title("Predicción de Evento Cardiovascular en Pacientes con VIH")
st.write("Introduce los datos del paciente para estimar el riesgo de ECV.")

# Entradas del usuario
sexo = st.selectbox("Sexo", ["Hombre", "Mujer"])
via_transmision = st.selectbox("Vía de transmisión", [
    "Homo/Bisexual", "Usuario de drogas inyectadas", "HSH+UDI", "Hemofílico",
    "Transfusión", "Contacto Heterosexual", "HSM+UDI", "Perinatal", "Otro"])
origen = st.selectbox("Origen", ["España", "No España"])
nivel_estudios = st.selectbox("Nivel de estudios", [
    "Sin estudios o primaria incompleta", "Primaria actual", "Secundaria Obligatoria",
    "Bachillerato", "Universitarios", "Otro", "Desconocido"])
sida = st.selectbox("Sida", ["No", "Sí", "Desconocido"])
edad = st.number_input("Edad", min_value=18, max_value=100, value=50)
carga_viral = st.selectbox("Carga viral al diagnóstico", ["<100.000 copias/ml", ">=100.000 copias/ml", "Desconocido"])
clase_tar = st.selectbox("Clase de TAR", ["1", "2", "3", "4"])
cd4 = st.number_input("CD4", min_value=0, max_value=2000, value=500)
anticore_vhb = st.selectbox("VHB anticore", ["Negativo", "Positivo", "Desconocido"])
serologia_vhc = st.selectbox("VHC", ["Negativo", "Positivo", "Desconocido"])
cociente_cd4_cd8 = st.number_input("Cociente CD4:CD8", min_value=0.0, max_value=5.0, value=1.0)
hta = st.selectbox("Hipertensión arterial", ["No HTA", "Si HTA"])
fumador = st.selectbox("Fumador", ["No fumador", "Si fumador"])
col_total = st.number_input("Colesterol total", min_value=0.0, value=180.0)
hdl = st.number_input("HDL", min_value=0.0, value=50.0)
trigliceridos = st.number_input("Triglicéridos", min_value=0.0, value=120.0)
col_no_hdl = st.number_input("Colesterol no HDL", min_value=0.0, value=130.0)
ratio_trig_hdl = st.number_input("Ratio Trig/HDL", min_value=0.0, value=2.4)
diabetes = st.selectbox("Diabetes", ["No", "Sí"])

# Codificación de variables
sexo = 1 if sexo == "Hombre" else 2
via_dict = {"Homo/Bisexual":1, "Usuario de drogas inyectadas":2, "HSH+UDI":3, "Hemofílico":4,
            "Transfusión":5, "Contacto Heterosexual":6, "HSM+UDI":7, "Perinatal":8, "Otro":90}
via_transmision = via_dict[via_transmision]
origen = 1 if origen == "España" else 2
sida_dict = {"No":0, "Sí":1, "Desconocido":9}
sida = sida_dict[sida]
edu_dict = {"Sin estudios o primaria incompleta":0, "Primaria actual":1, "Secundaria Obligatoria":2,
            "Bachillerato":3, "Universitarios":4, "Otro":8, "Desconocido":9}
nivel_estudios = edu_dict[nivel_estudios]
carga_dict = {"<100.000 copias/ml":1, ">=100.000 copias/ml":2, "Desconocido":3}
carga_viral = carga_dict[carga_viral]
vhb_dict = {"Negativo":0, "Positivo":1, "Desconocido":2}
anticore_vhb = vhb_dict[anticore_vhb]
vhc = vhb_dict[serologia_vhc]
hta = 1 if hta == "Si HTA" else 0
fumador = 1 if fumador == "Si fumador" else 0
diabetes = 1 if diabetes == "Sí" else 0

# Vector de entrada (ajustar orden y número según el modelo)
X_input = np.array([[sexo, via_transmision, origen, nivel_estudios, sida, edad, carga_viral,
                     int(clase_tar), cd4, 0, vhc, anticore_vhb, cociente_cd4_cd8, hta,
                     fumador, col_total, hdl, trigliceridos, col_no_hdl, ratio_trig_hdl, diabetes]])

# Predicción
if st.button("Predecir evento cardiovascular"):
    prob = modelo.predict_proba(X_input)[0][1]
    st.write(f"🔎 Probabilidad estimada de evento cardiovascular: **{prob:.2%}**")
