import pandas as pd
import os
from google.cloud.bigquery.client import Client
import streamlit as st


project_id = 'fiap-tech-challenge-4'
dataset_id = 'tech_challenge_4'
table_id = 'petroleo_brent'
client = Client(project = project_id)

try:
    query = f'SELECT * FROM {project_id}.{dataset_id}.{table_id} order by data desc;'

    query_job = client.query(query)

    df = query_job.to_dataframe()

except Exception as e:
    print(f'Ocorreu um erro ao obter os dados do Google BigQuery: {e}')


st.title('Petróleo Bruto Brent')

tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(["Introdução", "Insights", "Dashboard", "Predição do Preço", "Performance dos Modelos", "Detalhamento Técnico"])

with tab0:      

    st.markdown("## Introdução")

    paragrafo1_tab0 = "No dinâmico cenário do setor de energia, o petróleo Brent assume uma posição central como referência global para os preços do petróleo. Este projeto concentra-se na exploração e compreensão das nuances que moldam o comportamento do preço do petróleo Brent, utilizando dados do Instituto de Pesquisa Econômica Aplicada (<a href='https://www.example.com' target='_blank'>IPEA</a>). Além da análise dos dados, nossa abordagem vai além, incorporando a criação de um dashboard interativo que não apenas visualiza informações cruciais, mas também gera insights valiosos para a tomada de decisão no mercado de commodities."
    paragrafo2_tab0 = "A essência deste trabalho está na aplicação de modelos de séries temporais, uma ferramenta robusta na previsão de tendências em séries temporais. Nosso objetivo vai além de compreender o comportamento passado do preço do petróleo Brent; buscamos antecipar movimentos futuros, contribuindo para uma visão mais informada e estratégica. A escolha do modelo ideal resulta de uma análise meticulosa, assegurando que o método escolhido seja capaz de realizar previsões de maneira precisa e eficaz."
    paragrafo3_tab0 = "Além disso, avançamos na implementação prática dos resultados obtidos, desenvolvendo um ambiente de produção interativo por meio da plataforma Streamlit. Esse deploy em produção oferece uma aplicação prontamente acessível, permitindo que stakeholders e tomadores de decisão explorem visualmente os insights gerados, facilitando a incorporação dessas informações no processo decisório cotidiano."
    paragrafo4_tab0 = "Ao concluir este trabalho, não apenas contribuímos para uma compreensão mais profunda das dinâmicas que influenciam o preço do petróleo Brent, mas também fornecemos uma ferramenta prática e acionável para apoiar a tomada de decisão no complexo e volátil mercado de commodities."

    texto_justificado_com_link_no_primeiro = f"""
    <p style="text-align: justify;">{paragrafo1_tab0}</p>
    <p style="text-align: justify;">{paragrafo2_tab0}</p>
    <p style="text-align: justify;">{paragrafo3_tab0}</p>
    <p style="text-align: justify;">{paragrafo4_tab0}</p>
    """

    st.markdown(texto_justificado_com_link_no_primeiro, unsafe_allow_html=True)  

    

with tab1:
    '''
    ## Insights

    '''

    # Texto que você deseja justificar
    

    st.dataframe(df)

with tab2:
    '''
    ## Dashboard

    '''

with tab3:
    '''
    ## Predição do Preço Petróleo Brent  
    '''  

with tab4:
    '''
    ## Performance dos Modelos

    '''        

with tab5:
    '''
    ## Detalhamento Técnico

    '''