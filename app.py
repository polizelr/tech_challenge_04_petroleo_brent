import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from utils import RenameColumns, CastToFloat, CastToDatetime, FillMissingData, AddColumn, SetIndex, TransformIndexToColumn
from sklearn.pipeline import Pipeline
import joblib
from joblib import load
from datetime import datetime, timedelta
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import mean_squared_error
from statsforecast import StatsForecast
from statsforecast.models import (AutoARIMA, Naive, SeasonalExponentialSmoothingOptimized, 
                                  SeasonalNaive, SeasonalWindowAverage)
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from google.oauth2 import service_account
from google.cloud import bigquery

st.set_page_config(
    page_title="Petróleo Bruto Brent"
)

credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
client = bigquery.Client(credentials=credentials)

@st.cache_data(ttl=3600)
def run_query(query):
    query_job = client.query(query)
    df = query_job.to_dataframe()
    return df


project_id = 'fiap-tech-challenge-4'
dataset_id = 'tech_challenge_4'
table_id = 'petroleo_brent'

try:
    df = run_query(f'SELECT * FROM {project_id}.{dataset_id}.{table_id} order by data desc;')

except Exception as e:
    print(f'Ocorreu um erro ao obter os dados do Google BigQuery: {e}')


st.title('Petróleo Bruto Brent')

tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(["Introdução", "Insights", "Dashboard", "Predição do Preço", "Performance dos Modelos", "Detalhamento Técnico"])

with tab0:      

    st.markdown("## Introdução")

    paragrafo1_tab0 = "No dinâmico cenário do setor de energia, o petróleo Brent assume uma posição central como referência global para os preços do petróleo. Este projeto concentra-se na exploração e compreensão das nuances que moldam o comportamento do preço do petróleo Brent, utilizando dados do Instituto de Pesquisa Econômica Aplicada (<a href='http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view' target='_blank'>IPEA</a>). Além da análise dos dados, nossa abordagem vai além, incorporando a criação de um dashboard interativo que não apenas visualiza informações cruciais, mas também gera insights valiosos para a tomada de decisão no mercado de commodities."
    paragrafo2_tab0 = "A essência deste trabalho está na aplicação de modelos de séries temporais, uma ferramenta robusta na previsão de tendências em séries temporais. Nosso objetivo vai além de compreender o comportamento passado do preço do petróleo Brent; buscamos antecipar movimentos futuros, contribuindo para uma visão mais informada e estratégica. A escolha do modelo ideal resulta de uma análise meticulosa, assegurando que o método escolhido seja capaz de realizar previsões de maneira precisa e eficaz."
    paragrafo3_tab0 = "Além disso, avançamos na implementação prática dos resultados obtidos, desenvolvendo um ambiente de produção interativo por meio da plataforma Streamlit. Esse deploy em produção oferece uma aplicação prontamente acessível, permitindo que stakeholders e tomadores de decisão explorem visualmente os insights gerados, facilitando a incorporação dessas informações no processo decisório cotidiano."
    paragrafo4_tab0 = "Ao concluir este trabalho, não apenas contribuímos para uma compreensão mais profunda das dinâmicas que influenciam o preço do petróleo Brent, mas também fornecemos uma ferramenta prática e acionável para apoiar a tomada de decisão no complexo e volátil mercado de commodities."

    texto_justificado_tab0 = f"""
    <p style="text-align: justify;">{paragrafo1_tab0}</p>
    <p style="text-align: justify;">{paragrafo2_tab0}</p>
    <p style="text-align: justify;">{paragrafo3_tab0}</p>
    <p style="text-align: justify;">{paragrafo4_tab0}</p>
    """

    st.markdown(texto_justificado_tab0, unsafe_allow_html=True)  


with tab1:
  
    st.markdown("## Visão Geral dos Dados")

    df_tab1 = df.copy()

    pipeline_tab1 = Pipeline([
        ('rename_columns', RenameColumns()),         
        ('fill_missing_data', FillMissingData())
    ])

    df_pipe_tab1 = pipeline_tab1.transform(df_tab1) 

    paragrafo1_tab1 = "O gráfico abaixo apresenta o preço do barril do petróleo bruto Brent comercializado ao longo dos anos."

    texto_justificado1_tab1 = f'<p style="text-align: justify;">{paragrafo1_tab1}</p>'
    st.markdown(texto_justificado1_tab1, unsafe_allow_html=True) 

    fig = px.line(df_pipe_tab1, x='ds', y='y', title='Preço por Barril do Petróleo Bruto Brent')
    fig.update_xaxes(title='Data')
    fig.update_yaxes(title='Preço (US$)')

    st.plotly_chart(fig)
    

    st.markdown("## Crise Econômica Global de 2008")
    
    paragrafo2_tab1 = """
        Através da análise do gráfico, pode-se constatar um aumento acentuado no preço do barril do petróleo bruto Brent de 2007 até a primeira metade de 2008, atingindo o preço máximo histórico em julho de 2008 de US$ 143.95 dólares. Esse aumento no preço pode ser atribuído a uma combinação de fatores econômicos e geopolíticos, como exemplo:


        + **Demanda Global Crescente:** A forte demanda por petróleo, especialmente da China e de outros países em desenvolvimento, impulsionou os preços. O crescimento econômico global, especialmente nos setores industrial e de transporte, aumentou a necessidade de energia.

        + **Instabilidade Geopolítica:** Tensões geopolíticas em várias regiões produtoras de petróleo, como o Oriente Médio e o Norte da África, contribuíram para a incerteza quanto ao fornecimento. Eventos como a escalada de conflitos no Iraque, a instabilidade no Irã e a interrupção da produção na Nigéria impactaram as perspectivas de oferta, elevando os preços.

        + **Dólar Americano Fraco:** O enfraquecimento do dólar norte-americano também desempenhou um papel significativo. O petróleo é cotado em dólar, e quando o valor do dólar cai, os preços do petróleo tendem a subir para compensar a perda de valor da moeda.

        + **Restrições na Oferta:** Limitações na capacidade de produção da Organização dos Países Exportadores de Petróleo (OPEP) e a dificuldade em aumentar rapidamente a produção de petróleo em resposta à crescente demanda também exerceram pressão sobre os preços.
    """
    
    st.markdown(paragrafo2_tab1)

    
    paragrafo3_tab1 = """
        Contudo, esse aumento significativo no preço do barril do petróleo teve um impacto significativo na economia global, culminando, dentre outros fatores, na **Crise Econômica** que se desdobrou na segunda metade de 2008, devido a:

        + **Pressão sobre os Consumidores:** O aumento nos preços do petróleo resultou em custos mais elevados para os consumidores, especialmente no que diz respeito aos gastos com combustíveis e produtos derivados de petróleo. Isso reduziu a renda disponível das famílias, afetando negativamente o consumo e desencadeando uma desaceleração econômica.

        + **Inflação:** O aumento nos preços do petróleo contribuiu para a inflação, à medida que os custos de produção aumentaram para várias indústrias. Isso levou a um ambiente econômico em que os bancos centrais foram desafiados a equilibrar a necessidade de controlar a inflação e, ao mesmo tempo, estimular o crescimento econômico.

        + **Impacto nos Setores Sensíveis:** Setores intensivos em energia, como transporte, manufatura e agricultura, foram particularmente afetados pelos preços elevados do petróleo. Empresas nesses setores enfrentaram aumentos nos custos operacionais, levando a cortes de empregos e redução de investimentos.

        + **Desencadeamento de Crises Financeiras em Outros Setores:** O impacto inicial nos preços do petróleo desencadeou uma série de eventos que amplificaram a crise. As dificuldades enfrentadas por consumidores e empresas no pagamento de dívidas levaram a uma crise no mercado de hipotecas subprime nos Estados Unidos, que acabou se espalhando para o sistema financeiro global.

        + **Desaceleração Econômica Global:** A combinação de inflação, custos mais elevados e redução do consumo levou a uma desaceleração econômica global. Isso exacerbou os desafios enfrentados por instituições financeiras já sob pressão devido à crise do mercado imobiliário.    


        A Crise Econômica Global abalou os alicerces do mercado financeiro e esse fato reverberou nos preços do petróleo Brent, resultando em uma queda expressiva, conforme evidenciado abaixo:

    """
    st.markdown(paragrafo3_tab1)

    df_filtrado_tab1 = df_pipe_tab1[(df_pipe_tab1['ds'] > pd.to_datetime('2007-01-01')) & (df_pipe_tab1['ds'] < pd.to_datetime('2009-04-01'))].sort_values(by='ds')
       
    fig = px.line(df_filtrado_tab1, x='ds', y='y', title='Preço por Barril do Petróleo Bruto Brent')
    fig.update_xaxes(title='Data')
    fig.update_yaxes(title='Preço (US$)', range=[2, 150])
    fig.update_layout(width = 700)

    st.plotly_chart(fig)


    st.markdown("## Primavera Árabe")

    paragrafo4_tab1 = """
        A Primavera Árabe foi uma série de protestos, revoltas e manifestações populares que eclodiram em vários países do Oriente Médio e do Norte da África a partir de 2010. Esses eventos foram impulsionados por uma combinação de fatores, incluindo descontentamento político, social e econômico, além do desejo por reformas democráticas. Os países mais notoriamente afetados incluíram Tunísia, Egito, Líbia, Síria e Iêmen.

        A influência da Primavera Árabe no preço do petróleo Brent pode ser compreendida em vários pontos-chave:

        - **Instabilidade Geopolítica:** A Primavera Árabe desencadeou instabilidades geopolíticas significativas na região do Oriente Médio e Norte da África. Conflitos e incertezas políticas em países produtores de petróleo, como Líbia e Síria, afetaram as operações de produção e exportação de petróleo, gerando preocupações sobre a oferta global.

        - **Interrupções na Produção e Exportação:** Em países onde os protestos evoluíram para conflitos armados, como a Líbia, as interrupções na produção e exportação de petróleo contribuíram para a volatilidade nos preços do petróleo Brent. A oferta restrita em meio à demanda contínua resultou em picos nos preços do petróleo.

        - **Desafios para a OPEP:** A Organização dos Países Exportadores de Petróleo (OPEP) enfrentou desafios significativos durante a Primavera Árabe. A capacidade da OPEP para coordenar a produção e estabilizar os preços foi prejudicada pela instabilidade em alguns de seus membros-chave, afetando a influência do grupo sobre o mercado de petróleo.

        - **Impacto na Economia Global:** A instabilidade nos preços do petróleo Brent devido à Primavera Árabe teve implicações na economia global. A elevação dos preços do petróleo contribuiu para custos mais altos de produção e transporte em vários setores econômicos, impactando o crescimento econômico global.

        Ou seja, a Primavera Árabe influenciou os preços do petróleo Brent principalmente através da instabilidade geopolítica, interrupções na produção e exportação de petróleo e percepções de risco que levaram a especulações nos mercados financeiros. Esses fatores destacam a sensibilidade do mercado de petróleo a eventos geopolíticos em regiões-chave de produção.
    """
    
    st.markdown(paragrafo4_tab1)

    df_filtrado2_tab1 = df_pipe_tab1[(df_pipe_tab1['ds'] > pd.to_datetime('2010-12-01')) & (df_pipe_tab1['ds'] < pd.to_datetime('2012-12-31'))].sort_values(by='ds')
       
    fig = px.line(df_filtrado2_tab1, x='ds', y='y', title='Preço por Barril do Petróleo Bruto Brent Durante a Primavera Árabe')    
    fig.update_xaxes(title='Data')
    fig.update_yaxes(title='Preço (US$)', range=[2, 150])
    fig.update_layout(width = 800)
    st.plotly_chart(fig)


    st.markdown("## Expansão da Produção de Xisto nos EUA")

    paragrafo5_tab1 = """
        O petróleo de xisto refere-se a óleo extraído de forma não convencional a partir de formações de xisto, utilizando técnicas como fraturamento hidráulico (fracking). A exploração comercial em grande escala do petróleo de xisto começou nos Estados Unidos, onde as reservas significativas desses depósitos foram identificadas.

        A influência do petróleo de xisto nos preços do petróleo Brent pode ser compreendida em vários pontos-chave:

        - **Aumento da Oferta Global:** A exploração bem-sucedida de petróleo de xisto levou a um aumento significativo na oferta global de petróleo. Os Estados Unidos, em particular, transformaram-se de importadores líquidos para exportadores líquidos, alterando significativamente a dinâmica do mercado internacional de petróleo e reduzindo a dependência de algumas regiões produtoras tradicionais.

        - **Impacto na Dinâmica da OPEP:** O aumento da produção de petróleo de xisto desafiou a influência tradicional da Organização dos Países Exportadores de Petróleo (OPEP) sobre os preços e a oferta global. A OPEP viu sua capacidade de manipular os preços do petróleo enfraquecida, uma vez que o petróleo de xisto contribuiu para um mercado mais competitivo.

        - **Volatilidade nos Preços:** A rápida expansão da produção de petróleo de xisto contribuiu para maior volatilidade nos preços do petróleo Brent. A capacidade de resposta mais rápida da produção de xisto às mudanças nos preços tornou o mercado mais suscetível a flutuações, pois a oferta poderia aumentar ou diminuir rapidamente em resposta às condições do mercado.

        - **Pressão sobre Países Produtores Tradicionais:** Países tradicionalmente dependentes da exportação de petróleo, especialmente aqueles com custos de produção mais elevados, sentiram a pressão da crescente produção de xisto. Isso incluiu países da OPEP que dependiam de preços mais altos para sustentar suas economias.

        - **Inovação Tecnológica e Redução de Custos:** A exploração do petróleo de xisto foi impulsionada pela inovação tecnológica, resultando em processos mais eficientes e uma redução nos custos de produção ao longo do tempo. Isso teve um impacto direto na competitividade do petróleo de xisto em comparação com outras fontes de petróleo.

        - **Adaptação da Indústria Global:** A ascensão do petróleo de xisto obrigou a indústria global de energia a se adaptar a um novo paradigma. Empresas e países precisaram reavaliar suas estratégias de produção, investimentos e políticas energéticas à luz da mudança na dinâmica do mercado.

        Em resumo, o petróleo de xisto influenciou os preços do petróleo Brent ao aumentar a oferta global, desafiar o domínio da OPEP, introduzir maior volatilidade nos preços, pressionar países produtores tradicionais e estimular inovações tecnológicas na indústria de energia. Esses efeitos destacam a transformação significativa que o petróleo de xisto trouxe para o mercado global de petróleo.
    """
    
    st.markdown(paragrafo5_tab1)

    df_filtrado3_tab1 = df_pipe_tab1[(df_pipe_tab1['ds'] > pd.to_datetime('2010-01-01')) & (df_pipe_tab1['ds'] < pd.to_datetime('2019-12-31'))].sort_values(by='ds')
       
    fig = px.line(df_filtrado3_tab1, x='ds', y='y', title='Preço por Barril do Petróleo Bruto Brent Durante a Expansão da Produção de Xisto nos EUA')    
    fig.update_xaxes(title='Data')
    fig.update_yaxes(title='Preço (US$)', range=[2, 150])
    fig.update_layout(width = 800)
    st.plotly_chart(fig)


    st.markdown("## Pandemia Covid-19")

    paragrafo6_tab1 = """
        A pandemia de COVID-19, causada pelo vírus SARS-CoV-2, começou em dezembro de 2019 na cidade de Wuhan, na China, e rapidamente se disseminou globalmente. Em março de 2020, a Organização Mundial da Saúde (OMS) declarou a COVID-19 uma pandemia, desencadeando uma série de medidas de contenção, lockdowns e impactos significativos em diversos setores da sociedade.

        A pandemia teve um impacto substancial nos preços do petróleo Brent, e aqui estão alguns dos pontos-chave dessa influência:

        - **Colapso da Demanda por Energia:** Com lockdowns e restrições generalizadas em todo o mundo, a demanda global por energia despencou. Viagens aéreas, rodoviárias e marítimas foram drasticamente reduzidas, e setores industriais experimentaram paralisações, resultando em uma queda acentuada na demanda por petróleo.

        - **Excesso de Oferta e Armazenamento Cheio:** O colapso da demanda ocorreu em um contexto de uma oferta já elevada de petróleo, devido à guerra de preços entre a Arábia Saudita e a Rússia no início de 2020. A combinação de excesso de oferta e falta de capacidade de armazenamento levou a uma pressão extrema sobre os preços do petróleo Brent.

        - **Acordos de Produção da OPEP+:** Em resposta à crise, a Organização dos Países Exportadores de Petróleo (OPEP) e aliados, conhecidos como OPEP+, concordaram em reduzir a produção para equilibrar o mercado e sustentar os preços do petróleo. Esses acordos foram cruciais para conter a queda livre dos preços.

        - **Recuperação Gradual com a Vacinação:** A expectativa e o progresso na vacinação contra a COVID-19 começaram a alterar as perspectivas do mercado de petróleo. A antecipação de uma recuperação econômica impulsionou a demanda, levando a um aumento gradual nos preços do petróleo Brent.

        - **Volatilidade e Sensibilidade a Notícias Relacionadas à Pandemia:** Durante a pandemia, os preços do petróleo Brent tornaram-se extremamente sensíveis a desenvolvimentos relacionados à saúde pública e medidas de contenção. Notícias sobre avanços ou retrocessos na luta contra a COVID-19 influenciaram a confiança do mercado e os preços do petróleo.

        Dessa forma, a pandemia da COVID-19 teve um impacto significativo nos preços do petróleo Brent, causando uma queda abrupta devido ao colapso da demanda, excesso de oferta e problemas de armazenamento. As ações da OPEP+ e as expectativas de recuperação econômica com a vacinação contribuíram para a estabilização gradual dos preços ao longo do tempo.
    """
    
    st.markdown(paragrafo6_tab1)

    df_filtrado4_tab1 = df_pipe_tab1[(df_pipe_tab1['ds'] > pd.to_datetime('2019-01-01')) & (df_pipe_tab1['ds'] < pd.to_datetime('2022-06-01'))].sort_values(by='ds')
       
    fig = px.line(df_filtrado4_tab1, x='ds', y='y', title='Preço por Barril do Petróleo Bruto Brent Durante a Pandemia da Covid-19')    
    fig.update_xaxes(title='Data')
    fig.update_yaxes(title='Preço (US$)', range=[2, 150])
    fig.update_layout(width = 800)

    st.plotly_chart(fig)


with tab2: 

    st.markdown("## Dashboard")

    paragrafo1_tab2 = '<p style="text-align: justify;">Com o intuito de enriquecer as opções de análise dos dados referentes ao petróleo Brent, foi desenvolvido um dashboard no Power BI. Este instrumento proporciona uma visão abrangente da variação de preços ao longo dos anos, oferecendo aos usuários uma ferramenta eficaz para a interpretação e compreensão dessas informações.</p>'
    st.markdown(paragrafo1_tab2, unsafe_allow_html=True)

    paragrafo6_tab2 = "O dashboard desenvolvido encontra-se atualmente disponível para download e execução local em: <a href='https://github.com/polizelr/tech_challenge_04_petroleo_brent/blob/main/dashboards/Analise_Preco_Petroleo.pbix' target='_blank'>Dashboad Preço Petróleo Brent</a>. "

    texto_justificado_tab2 = f"""
        <p style="text-align: justify;">{paragrafo6_tab2}</p>    
    """
    st.markdown(texto_justificado_tab2, unsafe_allow_html=True)  


    st.markdown('### Conexão e Atualização de Dados')

    paragrafo2_tab2 = '<p style="text-align: justify;">O dashboard atualmente utiliza a opção de carga de dados e está conectado ao Big Query, assim como as demais partes do projeto. Recomenda-se, no entanto, como melhorias a publicação do projeto e a conexão via DirectQuery, a fim de otimizar o desempenho e a acessibilidade. A última carga de dados foi realizada em 25/01/2024, e é importante notar que o IPEA disponibilizou informações até 16/01/2024.</p>'
    st.markdown(paragrafo2_tab2, unsafe_allow_html=True)
    
    st.markdown('### Estrutura do Dashboard')

    paragrafo3_tab2 = '<p style="text-align: justify;">O design do dashboard é concebido com simplicidade e eficácia. Ao iniciar a navegação, o usuário é recebido por uma tela inicial, a "Home", que oferece opções para explorar as abas "Geral" e "Cenários".</p>'
    st.markdown(paragrafo3_tab2, unsafe_allow_html=True)

    st.image('./imagens/dashboard_home.png', caption='Estrutura do Dashboard')

    st.markdown('#### Aba Geral')

    paragrafo4_tab2 = '<p style="text-align: justify;">Na aba Geral, o usuário tem acesso a todos os dados disponíveis, com a capacidade de filtrar informações conforme o intervalo de tempo desejado. Esta seção destaca dois indicadores principais de Maior e Menor Valor, acompanhados por um gráfico de linha que representa a variação do preço ao longo do tempo. Além disso, uma tabela detalhada permite ao usuário analisar o preço do barril desde o nível anual até o diário. Optou-se pelo preço médio nesta tabela, pois essa agregação proporciona uma visão abrangente do comportamento médio dos preços em meses ou anos, destacando tendências relevantes.</p>'
    st.markdown(paragrafo4_tab2, unsafe_allow_html=True)

    st.image('./imagens/dashboard_geral.png', caption='Aba Geral')

    st.markdown('#### Aba Cenários')

    paragrafo5_tab2 = '<p style="text-align: justify;">Na aba Cenários, as mesmas análises da aba Geral estão disponíveis, mas agora é introduzido o conceito de cenários. O usuário pode filtrar diferentes cenários, atualmente suportando a Crise Econômica de 2008, a Primavera Árabe e a Pandemia de Covid-19. Ao utilizar o filtro de cenário, os dados são automaticamente ajustados para o período relevante, proporcionando insights específicos de cada situação. Essa funcionalidade não apenas facilita a usabilidade, mas também simplifica a construção e interpretação das análises, fornecendo uma perspectiva mais aprofundada.</p>'
    st.markdown(paragrafo5_tab2, unsafe_allow_html=True)

    st.image('./imagens/dashboard_cenarios.png', caption='Aba Cenários')


with tab3:
    
    st.markdown("## Predição do Preço Petróleo Brent")    

    df_tab3 = df.copy()

    pipeline_tab3 = Pipeline([
        ('rename_columns', RenameColumns()),        
        ('fill_missing_data', FillMissingData()),
        ('add_column', AddColumn())
    ])

    df_pipe_tab3 = pipeline_tab3.transform(df_tab3)

    input_days_to_predict = int(st.slider('Selecione quantos dias você quer predizer', 1, 30)) 

    if st.button('Enviar'):
        data_atual_tab3 = datetime.now().date()
        data_especifica = datetime.strptime('2024-01-22', '%Y-%m-%d').date()

        diferenca_em_dias = (data_atual_tab3 - data_especifica).days
        qtde_dias_a_predizer = input_days_to_predict + diferenca_em_dias


        model = joblib.load('modelo/sm.joblib')
        final_pred = model.predict(h = qtde_dias_a_predizer) 

        ultimo_dado_ipea = df_pipe_tab3['ds'].max()

        # transformações para melhorar a exibição dos dados na tabela
        final_pred_filtrado = final_pred[final_pred['ds'] > ultimo_dado_ipea] 
        final_pred_filtrado.rename(columns= {'ds' : 'Data', 'SeasWA': 'Preço Predito (US$)'}, inplace=True)
        final_pred_filtrado['Data'] = final_pred_filtrado['Data'].dt.normalize()  
        final_pred_filtrado.reset_index(drop=True, inplace=True)

        # transformações para que o gráfico exiba além do período predito, os dados dos 3 meses anteriores
        data_ha_3_meses = data_atual_tab3 - timedelta(days=3 * 30)

        df_pipe_tab3 = df_pipe_tab3[df_pipe_tab3['ds'] >= pd.to_datetime(data_ha_3_meses)]

        df_pipe_tab3.rename(columns= {'ds' : 'Data', 'y': 'Preço Predito (US$)'}, inplace=True)
        df_pipe_tab3_filtrado = df_pipe_tab3[['Data', 'Preço Predito (US$)']]   

        df_resultado = pd.concat([df_pipe_tab3_filtrado, final_pred_filtrado], ignore_index=True).sort_values(by='Data')        
        df_resultado.rename(columns= {'Preço Predito (US$)' : 'Preco_pretroleo_brent'}, inplace=True)
        df_resultado =df_resultado.reset_index(drop=True)

        fig = px.line(df_resultado, x=df_resultado['Data'], y=df_resultado['Preco_pretroleo_brent'], title='Previsão do Preço por Barril do Petróleo Bruto Brent')
        fig.update_xaxes(title='Data')
        fig.update_yaxes(title='Preço (US$)')

        
        # linha vertical tracejada alocada na data atual para marcar a transição entre o período do preço real e o período do preço predito
        fig.add_trace(go.Scatter(x=[data_atual_tab3, data_atual_tab3], y=[min(df_resultado['Preco_pretroleo_brent']), max(df_resultado['Preco_pretroleo_brent'])+2],
                         mode='lines',
                         line=dict(color='gray', dash='dash'),
                         name='Data atual'))
        

        #annotation para diferenciar o período do preço real do período do preço predito
        data_anterior = data_atual_tab3 - timedelta(days= 45)
        data_posterior = data_atual_tab3 + timedelta(days= input_days_to_predict/2)
        
        fig.add_annotation(
            x=data_anterior,
            y= max(df_resultado['Preco_pretroleo_brent']) + 1,
            text='Preço Real (US$)',
            showarrow=True,
            arrowhead=2,
            arrowcolor='green',
            arrowwidth=2,
            ax=0,
            ay=-40
        )

        fig.add_annotation(
            x=data_posterior,
            y= max(df_resultado['Preco_pretroleo_brent']) + 1,
            text='Preço Predito (US$)',
            showarrow=True,
            arrowhead=2,
            arrowcolor='red',
            arrowwidth=2,
            ax=0,
            ay=-40
        )        

        fig.update_layout(width = 1000)
 
        st.plotly_chart(fig)

        df_final_pred_filtrado = final_pred_filtrado[final_pred_filtrado['Data'] > datetime.now()].reset_index(drop=True)

        st.dataframe(df_final_pred_filtrado)


with tab4:    

    st.markdown("## Performance dos Modelos") 

    st.markdown("### Objetivo")

    st.markdown("O objetivo da construção deste modelo de machine learning é a predição do preço diário do barril de petróleo bruto Brent através da análise dos dados históricos.")

    st.markdown("### Base de Dados")

    paragrafo1_tab4 = "Para a realização desta análise, foram utilizados dados de periodicidade diária, obtidos no site do <a href='http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view' target='_blank'>IPEA</a>."

    paragrafo2_tab4 = "O IPEA, Instituto de Pesquisa Econômica Aplicada, é uma instituição brasileira vinculada ao governo federal. Ele foi criado em 1964 e tem como objetivo realizar pesquisas e estudos econômicos aplicados para subsidiar o governo na formulação e avaliação de políticas públicas."

    paragrafo3_tab4 = """
    A base de dados contém as seguintes informações:

    + **Data:** Data em que as informações foram registradas

    + **Preço Petróleo Bruto Brent:** preço em dólar do barril de petróleo bruto Brent

    """

    texto_justificado1_tab4 = f"""
    <p style="text-align: justify;">{paragrafo1_tab4}</p>
    <p style="text-align: justify;">{paragrafo2_tab4}</p>
    <p style="text-align: justify;">{paragrafo3_tab4}</p>  
    """

    st.markdown(texto_justificado1_tab4, unsafe_allow_html=True)

    st.markdown("### Transformação de Dados")

    st.markdown(f'<p style="text-align: justify;">Para a transformação, foi realizada a leitura dos dados da tabela "{table_id}" armazenada no Google Big Query e as colunas foram ajustadas para o padrão de time series.</p>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: justify;">Devido a falta de dados nos finais de semana e feriados, por não haver comercialização de petróleo nestes dias, o valor do fechamento do último dia útil disponível será utilizado para preencher esses gaps, uma vez que os modelos de séries temporais necessitam que a periodicidade seja uniforme durante todo o período.</p>', unsafe_allow_html=True)

    df_tab4 = df.copy()

    pipeline_tab4 = Pipeline([
        ('rename_columns', RenameColumns()),        
        ('cast_to_datetime', CastToDatetime()),  
        ('fill_missing_data', FillMissingData()),
        ('set_index', SetIndex())
    ])

    df_pipe_tab4 = pipeline_tab4.transform(df_tab4)
    
    st.dataframe(df_pipe_tab4)


    st.markdown("### Seasonal Decompose")

    paragrafo4_tab4 = "Após a finalização das transformações dos dados, podemos aplicar e analisar o seasonal decompose."
    paragrafo5_tab4 = "O seasonal decompose divide uma série em três principais componentes: tendência, sazonalidade e resíduos (o primeiro gráfico mostra os dados da série em si)."

    texto_justificado2_tab4 = f"""
    <p style="text-align: justify;">{paragrafo4_tab4}</p>
    <p style="text-align: justify;">{paragrafo5_tab4}</p>    
    """

    st.markdown(texto_justificado2_tab4, unsafe_allow_html=True)


    st.markdown("#### Tendência (Trend)")

    paragrafo6_tab4 = '<p style="text-align: justify;">A tendência representa a direção geral dos dados ao longo do tempo. E ela é crucial para entender o comportamento de longo prazo dos dados e pode ser usada para fazer previsões de longo prazo.</p>'
    st.markdown(paragrafo6_tab4, unsafe_allow_html=True)
    
    st.markdown("#### Sazonalidade (Seasonal)")

    paragrafo7_tab4 = '<p style="text-align: justify;">Já a sazonalidade se refere a padrões repetitivos em intervalos regulares de tempo, como sazonalidade diária, mensal ou anual. A sazonalidade é importante para entender os padrões cíclicos e prever eventos sazonais, como vendas sazonais ou comportamento do mercado.</p>'
    st.markdown(paragrafo7_tab4, unsafe_allow_html=True)

    st.markdown("#### Resíduos (Residual)")
    paragrafo8_tab4 = '<p style="text-align: justify;">Os resíduos são a parte não explicada pela tendência e pela sazonalidade. Eles representam o "ruído" ou a aleatoriedade na série e são importantes para entender a variação não explicada na série. Modelos de previsão frequentemente se concentram na modelagem dos resíduos para melhorar a precisão das previsões. A análise deste resultado é fundamental para entender a estrutura e os componentes da série.</p>'
    st.markdown(paragrafo8_tab4, unsafe_allow_html=True)

    
    results = seasonal_decompose(df_pipe_tab4)  
   
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(40, 20))

    # Gráfico 1 - Observado
    ax1.plot(results.observed)
    ax1.set_title("Observado")
    ax1.set_xlabel("Data")
    ax1.set_ylabel("Valor Observado")

    # Gráfico 2 - Tendência
    ax2.plot(results.trend)
    ax2.set_title("Tendência")
    ax2.set_xlabel("Data")
    ax2.set_ylabel("Valor da Tendência")

    # Gráfico 3 - Sazonalidade
    ax3.plot(results.seasonal)
    ax3.set_title("Sazonalidade")
    ax3.set_xlabel("Data")
    ax3.set_ylabel("Valor da Sazonalidade")

    # Gráfico 4 - Resíduos
    ax4.plot(results.resid)
    ax4.set_title("Resíduos")
    ax4.set_xlabel("Data")
    ax4.set_ylabel("Valor dos Resíduos")

   
    plt.tight_layout()    
    st.pyplot(fig)

    paragrafo9_tab4 = """
    <p style="text-align: justify;">
    A partir dos gráficos acima, pode-se constatar que:

    **Análise da Tendência**

    A tendência é crescente, indicando um aumento ao longo do tempo, contudo, não pode ser definido como um aumento constante.


    **Análise da Sazonalidade**

    Não há uma definição de períodos de sazonalidade.


    **Análise dos Resíduos**

    Os resíduos variam bastante ao longo do tempo, com uma certa tendencia de crescimento e com alguns picos bem definidos.
    </p>
    """
    st.markdown(paragrafo9_tab4, unsafe_allow_html=True)

    st.markdown("### Augmented Dickey-Fuller (ADF) e Transformações")

    st.markdown('<p style="text-align: justify;">O Augmented Dickey-Fuller (ADF) é um teste estatístico utilizado para avaliar a estacionariedade em séries temporais. A estacionariedade é uma propriedade importante, pois muitos modelos de análise de séries temporais assumem que os dados sejam estacionários. A estacionariedade implica que as propriedades estatísticas da série, como a média e a variância, permaneçam constantes ao longo do tempo. Como resultado, o ADF retorna um valor de teste estatístico um p-value.</p>', unsafe_allow_html=True)
    st.markdown("- **Hipótese Nula (H0):** a hipótese nula assume que a série temporal é não estacionária. Ou seja, ela possui raiz unitária, o que significa que a série possui alguma forma de tendência ou estrutura não estacionária.")
    st.markdown("- **Hipótese Alternativa (H1):** a hipótese alternativa é a oposta da hipótese nula. Se o p-value for suficientemente baixo, você rejeita a hipótese nula e conclui que a série é estacionária.")
    st.markdown("Analisando o resultado:")
    st.markdown("- Se o p-value for menor que o nível de significância adotado (0,05) a hipótese nula é rejeitada e conclui-se que a série é estacionária. Além disso, também vamos analisar se o valor do teste estatísico é menor que o valor crítico de 5%")
    st.markdown("- Se o valor-p for maior que o nível de significância, a hipótese nula não pode ser rejeitada, o que sugere que a série é não estacionária.")

    X = df_pipe_tab4.y.values
    adf_result = adfuller(X)

    st.markdown("#### Teste ADF")
    st.write(f"**Teste Estatístico**: {adf_result[0]}")
    st.write(f"**P-Value**: {adf_result[1]}")
    st.write("**Valores críticos:**")

    for key, value in adf_result[4].items():
      st.write(f"\t**{key}**: {value}")
    
    st.markdown("De acordo com as regras descritas acima, pode-se confirmar que a série não é estacionária.")
    st.markdown("Também podemos confirmar a não estacionaridade através da análise gráfica.")

    ma = df_pipe_tab4.rolling(12).mean()

    fig, ax = plt.subplots()
    df_pipe_tab4.plot(ax=ax, legend=False)
    ma.plot(ax=ax, legend=False, color='r')
    ax.set_xlabel('Data')
    ax.set_ylabel('Preço (US$)')
    ax.set_title('Preço por Barril do Petróleo Bruto Brent')
    st.pyplot(fig)

    st.markdown("Para transformar a série em estacionária, primeiramente, vamos aplicar a função de log.")

    df_preco_petroleo_log = np.log(df_pipe_tab4)
    ma_log = df_preco_petroleo_log.rolling(12).mean()

    fig, ax = plt.subplots()
    df_preco_petroleo_log.plot(ax=ax, legend=False)
    ma_log.plot(ax=ax, legend=False, color='r')
    ax.set_xlabel('Data')
    ax.set_ylabel('Preço (US$)')
    ax.set_title('Preço por Barril do Petróleo Bruto Brent - Escala Logarítmica')
    st.pyplot(fig)

    st.markdown("Através do gráfico acima, pode-se constatar que a escala foi ajustada, porém ainda não aparenta ser uma série estacionária. Aplicaremos novamente o ADF para comprovar")

    X_log = df_preco_petroleo_log.y.values
    adf_result_log = adfuller(X_log)

    st.markdown("#### Teste ADF")
    st.write(f"**Teste Estatístico**: {adf_result_log[0]}")
    st.write(f"**P-Value**: {adf_result_log[1]}")
    st.write("**Valores críticos:**")

    for key, value in adf_result_log[4].items():
      st.write(f"\t**{key}**: {value}")


    st.markdown("Através dos resultados acima, confirmamos que a série ainda não é estacionária. Então, vamos aplicar a subtração do valor pela média móvel.")

    df_preco_petroleo_s = (df_preco_petroleo_log - ma_log).dropna()

    ma_s = df_preco_petroleo_s.rolling(12).mean()

    std = df_preco_petroleo_s.rolling(12).std()

    fig, ax = plt.subplots()
    df_preco_petroleo_s.plot(ax=ax, legend=False)
    ma_s.plot(ax=ax, legend=False, color='r')
    std.plot(ax=ax, legend=False, color='g')
    ax.set_xlabel('Data')
    ax.set_ylabel('Preço (US$)')
    ax.set_title('Preço por Barril do Petróleo Bruto Brent -  Série Estacionária')
    st.pyplot(fig)

    X_s = df_preco_petroleo_s.y.values
    adf_result_s = adfuller(X_s)

    st.markdown("#### Teste ADF")
    st.write(f"**Teste Estatístico**: {adf_result_s[0]}")
    st.write(f"**P-Value**: {adf_result_s[1]}")
    st.write("**Valores críticos:**")

    for key, value in adf_result_s[4].items():
      st.write(f"\t**{key}**: {value}")
    
    st.markdown("Confirmamos dessa forma que a série se tornou estacionária.")

    st.markdown("### ACF e PACF")

    paragrafo30_tab4 = '<p style="text-align: justify;">O ACF (AutoCorrelation Function) e o PACF (Partial AutoCorrelation Function) são ferramentas essenciais na análise de séries temporais. Eles são usados para entender a estrutura de dependência temporal nos dados e são fundamentais para a modelagem e previsão de séries temporais.</p>'
    st.markdown(paragrafo30_tab4, unsafe_allow_html=True)

    st.markdown("**ACF**")
    st.markdown("- Quanto um período está relacionado com o outro, direta e indiretamente")
    st.markdown("**PACF**")
    st.markdown("- Quanto um período está relacionado apenas diretamente")
    
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    sm.graphics.tsa.plot_acf(df_preco_petroleo_s['y'], lags=40, ax=ax[0])   
    sm.graphics.tsa.plot_pacf(df_preco_petroleo_s['y'], lags=40, ax=ax[1])
    
    ax[0].set_title('Autocorrelation Function (ACF)')
    ax[1].set_title('Partial Autocorrelation Function (PACF)')    
    st.pyplot(fig)    
    
    st.markdown("Embora seja uma função composta, há uma relação forte do preço anterior do barril de petróleo bruto Brent com o próximo.")


    st.markdown("### Métricas de Avaliação")

    st.markdown("As métricas de avaliação utilizadas para medir o desempenho dos modelos são:")

    st.markdown("#### WMAPE")

    paragrafo10_tab4 = "O WMAPE (Weighted Mean Absolute Percentage Error) é uma métrica de avaliação usada na análise de séries temporais e previsão. Ele mede a precisão de um modelo de previsão em relação aos valores reais, levando em consideração pesos, caso necessário."
    paragrafo11_tab4 = "Uma pontuação mais baixa de WMAPE indica maior precisão na previsão, mas a interpretação depende do contexto e da relevância dos dados."

    texto_justificado3_tab4 = f"""
    <p style="text-align: justify;">{paragrafo10_tab4}</p>
    <p style="text-align: justify;">{paragrafo11_tab4}</p>    
    """
    st.markdown(texto_justificado3_tab4, unsafe_allow_html=True)

    def wmape(y_true, y_pred):
        return np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()
    

    st.markdown("#### Erro Médio Percentual Absoluto (MAPE)")

    paragrafo12_tab4 = "O MAPE (Mean Absolute Percentage Error), é uma medida usada para avaliar a precisão de previsões em relação aos valores reais. Ele calcula a média das diferenças absolutas entre as previsões e os valores reais e é expresso em porcentagem."
    paragrafo13_tab4 = "Quanto menor o valor do MAPE, melhor é a precisão do modelo. Isso indica que as previsões estão, em média, mais próximas dos valores reais. Enquanto um valor alto, indica que as previsões estão longe dos valores observados e que o modelo precisa ser revisto."

    texto_justificado4_tab4 = f"""
    <p style="text-align: justify;">{paragrafo12_tab4}</p>
    <p style="text-align: justify;">{paragrafo13_tab4}</p>    
    """
    st.markdown(texto_justificado4_tab4, unsafe_allow_html=True)
    
    def mape(y_true, y_pred):
        return np.mean(np.abs((y_true.values - y_pred.values)/ y_true.values))


    st.markdown("#### Root Mean Square Error (RMSE)")

    paragrafo14_tab4 = "O RMSE (Root Mean Square Error) é uma métrica comum de avaliação em análise de séries temporais e previsão. Ele mede o erro médio quadrático entre as previsões e os valores reais, proporcionando uma medida da precisão do modelo."
    paragrafo15_tab4 = "Quanto menor o valor do RMSE, maior a precisão da previsão. No entanto, sua interpretação deve levar em conta a escala dos dados e o contexto do problema."

    texto_justificado5_tab4 = f"""
    <p style="text-align: justify;">{paragrafo14_tab4}</p>
    <p style="text-align: justify;">{paragrafo15_tab4}</p>    
    """
    st.markdown(texto_justificado5_tab4, unsafe_allow_html=True)

    def rmse(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        return rmse


    st.markdown("### Statsforecast")

    paragrafo16_tab4 = '<p style="text-align: justify;">O StatsForecast é um pacote em Python que oferece funcionalidades para modelagem e previsão de séries temporais.</p>'
    st.markdown(paragrafo16_tab4, unsafe_allow_html=True)
    
    st.markdown("#### Formatação dos Dados")

    paragrafo17_tab4 = '<p style="text-align: justify;">Para utilizar o StatsForecast, é necessário fornecer como entrada um dataframe com as colunas <b>ds</b>, <b>y</b> e <b>unique_id</b>, em que <b>ds</b> representa a data associada a cada ponto dos dados da série temporal, <b>y</b> os valores numéricos e <b>unique_id</b> uma chave única.</p>'
    st.markdown(paragrafo17_tab4, unsafe_allow_html=True)

    pipeline_tab4_sf = Pipeline([
        ('rename_columns', RenameColumns()),        
        ('cast_to_datetime', CastToDatetime()),  
        ('fill_missing_data', FillMissingData()),
        ('add_column', AddColumn())
    ])

    df_pipe_tab4_sf = pipeline_tab4_sf.transform(df_tab4) 
    st.dataframe(df_pipe_tab4_sf)

    st.markdown("#### Modelagem")

    paragrafo18_tab4 = "Para iniciar a modelagem, dividiremos os dados em dois grupos: treino e teste."
    paragrafo19_tab4 = "Dividir os dados em treino e teste no contexto de séries temporais é uma boa prática porque reflete a natureza do problema de previsão temporal. A ordem temporal dos dados é fundamental em séries temporais, pois a informação do passado é usada para prever o futuro. Dividir os dados em treino e teste permite simular essa situação realista, treinando o modelo em dados anteriores e avaliando sua capacidade de prever dados futuros não vistos. Isso ajuda a verificar se o modelo é capaz de generalizar e fazer previsões precisas em cenários reais, evitando vazamento de informações do futuro para o passado."

    texto_justificado6_tab4 = f"""
    <p style="text-align: justify;">{paragrafo18_tab4}</p>
    <p style="text-align: justify;">{paragrafo19_tab4}</p>    
    """
    st.markdown(texto_justificado6_tab4, unsafe_allow_html=True)

    treino = df_pipe_tab4_sf[(df_pipe_tab4_sf['ds'] >= pd.to_datetime('2022-10-01')) & (df_pipe_tab4_sf['ds'] < pd.to_datetime('2023-10-01'))]
    valid = df_pipe_tab4_sf[(df_pipe_tab4_sf['ds'] >= pd.to_datetime('2023-10-01')) & (df_pipe_tab4_sf['ds'] < pd.to_datetime('2024-01-22'))]

    h = valid.index.nunique()

    st.markdown("#### Execução dos Modelos")

    st.markdown("##### Naive")

    paragrafo20_tab4 = '<p style="text-align: justify;">Para iniciarmos a análise do StatsForecast, vamos utilizar o modelo básico Naive que é uma abordagem simples em que a previsão para o próximo período é igual à última observação conhecida.</p>'
    st.markdown(paragrafo20_tab4, unsafe_allow_html=True)

    model = StatsForecast(models=[Naive()], freq='D', n_jobs=-1)
    model.fit(treino)

    forecast_naive = model.predict(h=h, level=[90])
    forecast_naive = forecast_naive.reset_index().merge(valid, on=['ds', 'unique_id'], how='left')

    wmape1 = wmape(forecast_naive['y'].values, forecast_naive['Naive'].values)
    rmse1 = rmse(forecast_naive['y'].values, forecast_naive['Naive'].values)
    mape1 = mape(forecast_naive['y'], forecast_naive['Naive'])    

    df_pipe_tab4_sf_filtered = df_pipe_tab4_sf[df_pipe_tab4_sf['ds'] >= pd.to_datetime('2023-06-01')]
     
    fig = go.Figure()

    # série temporal prevista e intervalo de confiança
    fig.add_trace(go.Scatter(x=forecast_naive['ds'], y=forecast_naive['Naive'], mode='lines', name='Previsão (Naive)', line=dict(color='rgb(101, 101, 101)')))
    fig.add_trace(go.Scatter(x=forecast_naive['ds'], y=forecast_naive['Naive-lo-90'], fill=None, mode='lines', line=dict(color='rgb(228, 228, 228)', width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=forecast_naive['ds'], y=forecast_naive['Naive-hi-90'], fill='tonexty', mode='lines', line=dict(color='rgb(228, 228, 228)', width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=df_pipe_tab4_sf_filtered['ds'], y=df_pipe_tab4_sf_filtered['y'], mode='lines', name='Valor Real', line=dict(color='blue')))              
  
    fig.update_layout(title='Previsão do Preço (US$) do Petróleo Brent Utilizando o Modelo Naive', xaxis_title='Data', yaxis_title='Preço (US$)', width=1000)   
    st.plotly_chart(fig)

    st.write("**Resultados:**")
    st.write(f"**WMAPE**: {wmape1:.2%}")
    st.write(f"**RMSE**: {rmse1:.2f}")
    st.write(f"**MAPE**: {mape1:.2%}")

    paragrafo20_tab4 = '<p style="text-align: justify;">Podemos constatar que o modelo não obteve um bom resultado, pois a previsão realizada através do último valor conhecido, não captura o comportamento dos dados.</p>'
    st.markdown(paragrafo20_tab4, unsafe_allow_html=True)
    

    st.markdown("##### Seasonal Naive")

    paragrafo21_tab4 = '<p style="text-align: justify;">O Seasonal Naive é um modelo semelhante ao Naive, mas é aplicado a séries temporais sazonais. A previsão para o próximo período é igual à observação do mesmo período na temporada anterior.</p>'
    st.markdown(paragrafo21_tab4, unsafe_allow_html=True)

    paragrafo21_tab4 = '<p style="text-align: justify;">Para buscar o melhor resultado possível, vamos utilizar uma técnica para encontrar o melhor valor de season_length possível.</p>'
    st.markdown(paragrafo21_tab4, unsafe_allow_html=True)

    best_mape_seasonal_naive = np.inf
    best_season_length_seasonal_naive = None

    # Grid Search for SeasonalNaive
    # Parâmetro já calculado previamente, para fornecer melhor performance a aplicação streamlit
    for season_length in [346]:
        model_s_1 = StatsForecast(models=[SeasonalNaive(season_length=season_length)], freq='D', n_jobs=-1)
        model_s_1.fit(treino)

        # Make predictions on validation set
        forecast_seasonal_naive_1 = model_s_1.predict(h=h,  level=[90])
        forecast_seasonal_naive_1 = forecast_seasonal_naive_1.reset_index().merge(valid, on=['ds', 'unique_id'], how='left')

        # Calculate MAPE
        wmape99 = wmape(forecast_seasonal_naive_1['y'].values, forecast_seasonal_naive_1['SeasonalNaive'].values)

        # Update best parameters if necessary
        if wmape99 < best_mape_seasonal_naive:
            best_mape_seasonal_naive = wmape99
            best_season_length_seasonal_naive = season_length
    
    st.write("**Resultado:**")
    st.write("Best Season Length:", best_season_length_seasonal_naive)

    paragrafo22_tab4 = f'<p style="text-align: justify;">O melhor Season Length foi de {best_season_length_seasonal_naive} para os parâmetros de treino utilizado.</p>'
    st.markdown(paragrafo22_tab4, unsafe_allow_html=True)

    model_s = StatsForecast(models=[SeasonalNaive(season_length=best_season_length_seasonal_naive)], freq='D', n_jobs=-1)
    fitted_model = model_s.fit(treino)

    forecast_seasonal_naive = model_s.predict(h=h, level=[90])
    forecast_seasonal_naive = forecast_seasonal_naive.reset_index().merge(valid, on=['ds', 'unique_id'], how='left')

    wmape2 = wmape(forecast_seasonal_naive['y'].values, forecast_seasonal_naive['SeasonalNaive'].values)
    rmse2 = rmse(forecast_seasonal_naive['y'].values, forecast_seasonal_naive['SeasonalNaive'].values)
    mape2 = mape(forecast_seasonal_naive['y'], forecast_seasonal_naive['SeasonalNaive'])    

    fig = go.Figure()

    # série temporal prevista e intervalo de confiança
    fig.add_trace(go.Scatter(x=forecast_seasonal_naive['ds'], y=forecast_seasonal_naive['SeasonalNaive'], mode='lines', name='Previsão (Seasonal Naive)', line=dict(color='rgb(101, 101, 101)')))
    fig.add_trace(go.Scatter(x=forecast_seasonal_naive['ds'], y=forecast_seasonal_naive['SeasonalNaive-lo-90'], fill=None, mode='lines', line=dict(color='rgb(228, 228, 228)', width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=forecast_seasonal_naive['ds'], y=forecast_seasonal_naive['SeasonalNaive-hi-90'], fill='tonexty', mode='lines', line=dict(color='rgb(228, 228, 228)', width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=df_pipe_tab4_sf_filtered['ds'], y=df_pipe_tab4_sf_filtered['y'], mode='lines', name='Valor Real', line=dict(color='blue')))              
  
    fig.update_layout(title='Previsão do Preço (US$) do Petróleo Brent Utilizando o Modelo Seasonal Naive', xaxis_title='Data', yaxis_title='Preço (US$)', width=1000)   
    st.plotly_chart(fig)

    st.write("**Resultados:**")
    st.write(f"**WMAPE**: {wmape2:.2%}")
    st.write(f"**RMSE**: {rmse2:.2f}")
    st.write(f"**MAPE**: {mape2:.2%}")

    paragrafo23_tab4 = f'<p style="text-align: justify;">Ao utilizar o Seasonal Naive, percebemos que o WMAPE: {wmape2:.2%}, RMSE: {rmse2:.2f} e MAPE: {mape2:.2%} obtiveram resultados melhores, além do fato dele indicar uma previsão mais próxima dos dados reais, quando comparado com os resultados obtidos no modelo Naive (WMAPE: {wmape1:.2%}, RMSE: {rmse1:.2f} e MAPE: {mape1:.2%}).</p>'
    st.markdown(paragrafo23_tab4, unsafe_allow_html=True)

    st.markdown("##### Seasonal Window Average")

    paragrafo24_tab4 = '<p style="text-align: justify;">Este modelo usa uma média móvel para suavizar as flutuações sazonais nos dados. A previsão é obtida tirando a média dos valores nas janelas temporais correspondentes na temporada anterior.</p>'
    st.markdown(paragrafo24_tab4, unsafe_allow_html=True)

    paragrafo28_tab4 = '<p style="text-align: justify;">Para buscar o melhor resultado possível, vamos utilizar uma técnica para encontrar o melhor valor de season_length e window_size possível.</p>'
    st.markdown(paragrafo28_tab4, unsafe_allow_html=True)

    best_mape_swa = np.inf
    best_season_length_swa = None
    best_window_size_swa = None

    # Grid Search for SeasonalWindowAverage
    # Parâmetros já calculados previamente, para fornecer melhor performance a aplicação streamlit
    for season_length in [181]:
        for window_size in [2]:
            model_sm_1 = StatsForecast(models=[SeasonalWindowAverage(season_length=season_length, window_size=window_size)], freq='D', n_jobs=-1)
            model_sm_1.fit(treino)

            # Make predictions on validation set
            forecast_dfsm_1 = model_sm_1.predict(h=h)
            forecast_dfsm_1 = forecast_dfsm_1.reset_index().merge(valid, on=['ds', 'unique_id'], how='left')

            # Calculate MAPE
            wmape98 = wmape(forecast_dfsm_1['y'].values, forecast_dfsm_1['SeasWA'].values)

            # Update best parameters if necessary
            if wmape98 < best_mape_swa:
                best_mape_swa = wmape98
                best_season_length_swa = season_length
                best_window_size_swa = window_size
    
    st.write("**Resultados:**")
    st.write("Best Season Length:", best_season_length_swa)
    st.write("Best Window Size:", best_window_size_swa)

    model_sm = StatsForecast(models=[SeasonalWindowAverage(season_length=best_season_length_swa, window_size=best_window_size_swa)], freq='D', n_jobs=-1) 
    model_sm.fit(treino)

    forecast_dfsm = model_sm.predict(h=h)
    forecast_dfsm = forecast_dfsm.reset_index().merge(valid, on=['ds', 'unique_id'], how='left')

    wmape3 = wmape(forecast_dfsm['y'].values, forecast_dfsm['SeasWA'].values)
    rmse3 = rmse(forecast_dfsm['y'].values, forecast_dfsm['SeasWA'].values)
    mape3 = mape(forecast_dfsm['y'], forecast_dfsm['SeasWA'])

    
    fig = go.Figure()

    # série temporal prevista e intervalo de confiança
    fig.add_trace(go.Scatter(x=forecast_dfsm['ds'], y=forecast_dfsm['SeasWA'], mode='lines', name='Previsão (SeasWA)', line=dict(color='rgb(101, 101, 101)')))    
    fig.add_trace(go.Scatter(x=df_pipe_tab4_sf_filtered['ds'], y=df_pipe_tab4_sf_filtered['y'], mode='lines', name='Valor Real', line=dict(color='blue')))              
  
    fig.update_layout(title='Previsão do Preço (US$) do Petróleo Brent Utilizando o Modelo Seasonal Window Average', xaxis_title='Data', yaxis_title='Preço (US$)', width=1000)   
    st.plotly_chart(fig)

    st.write("**Resultados:**")
    st.write(f"**WMAPE**: {wmape3:.2%}")
    st.write(f"**RMSE**: {rmse3:.2f}")
    st.write(f"**MAPE**: {mape3:.2%}")

    paragrafo25_tab4 = f'<p style="text-align: justify;">Utilizando a sazonalidade de {best_season_length_swa} e a janela de {best_window_size_swa}, o Seasonal Window Average obteve resultados excelentes (WMAPE: {wmape3:.2%}, RMSE: {rmse3:.2f} e MAPE: {mape3:.2%}), quando comparado aos modelos anteriores, além de conseguir capturar bem o comportamento dos dados.</p>'
    st.markdown(paragrafo25_tab4, unsafe_allow_html=True)


    st.markdown("##### Seasonal Exponential Smoothing Optimized")

    paragrafo26_tab4 = '<p style="text-align: justify;">A suavização exponencial é uma técnica que atribui pesos decrescentes às observações passadas. A versão sazonal otimizada estende isso para capturar padrões sazonais. É um método mais avançado que ajusta automaticamente os parâmetros para otimizar a precisão da previsão.</p>'
    st.markdown(paragrafo26_tab4, unsafe_allow_html=True)

    paragrafo29_tab4 = '<p style="text-align: justify;">Para buscar o melhor resultado possível, vamos utilizar uma técnica para encontrar o melhor valor de season_length possível.</p>'
    st.markdown(paragrafo29_tab4, unsafe_allow_html=True)

    best_mape_seso = np.inf
    best_season_length_seso = None

    # Grid Search for SeasonalExponentialSmoothingOptimized
    # Parâmetro já calculado previamente, para fornecer melhor performance a aplicação streamlit
    for season_length in [171]:
        model_seso_1 = StatsForecast(models=[SeasonalExponentialSmoothingOptimized(season_length=season_length)], freq='D', n_jobs=-1)
        model_seso_1.fit(treino)

        # Make predictions on validation set
        forecast_seasonal_exp_smo_opt_1 = model_seso_1.predict(h=h)
        forecast_seasonal_exp_smo_opt_1 = forecast_seasonal_exp_smo_opt_1.reset_index().merge(valid, on=['ds', 'unique_id'], how='left')

        # Calculate MAPE
        wmape97 = wmape(forecast_seasonal_exp_smo_opt_1['y'].values, forecast_seasonal_exp_smo_opt_1['SeasESOpt'].values)

        # Update best parameters if necessary
        if wmape97 < best_mape_seso:
            best_mape_seso = wmape97
            best_season_length_seso = season_length
            best_window_size = window_size

    st.write("Best Season Length:", best_season_length_seso)

    model_seso = StatsForecast(models=[SeasonalExponentialSmoothingOptimized(season_length=best_season_length_seso)], freq='D', n_jobs=-1)
    model_seso.fit(treino)

    forecast_seasonal_exp_smo_opt = model_seso.predict(h=h)
    forecast_seasonal_exp_smo_opt = forecast_seasonal_exp_smo_opt.reset_index().merge(valid, on=['ds', 'unique_id'], how='left')

    wmape4 = wmape(forecast_seasonal_exp_smo_opt['y'].values, forecast_seasonal_exp_smo_opt['SeasESOpt'].values)
    rmse4 = rmse(forecast_seasonal_exp_smo_opt['y'].values, forecast_seasonal_exp_smo_opt['SeasESOpt'].values)
    mape4 = mape(forecast_seasonal_exp_smo_opt['y'], forecast_seasonal_exp_smo_opt['SeasESOpt'])

    fig = go.Figure()

    # série temporal prevista e intervalo de confiança
    fig.add_trace(go.Scatter(x=forecast_seasonal_exp_smo_opt['ds'], y=forecast_seasonal_exp_smo_opt['SeasESOpt'], mode='lines', name='Previsão (SeasESOpt)', line=dict(color='rgb(101, 101, 101)')))    
    fig.add_trace(go.Scatter(x=df_pipe_tab4_sf_filtered['ds'], y=df_pipe_tab4_sf_filtered['y'], mode='lines', name='Valor Real', line=dict(color='blue')))              
  
    fig.update_layout(title='Previsão do Preço (US$) do Petróleo Brent Utilizando o Modelo Seasonal Exponential Smoothing Optimized', xaxis_title='Data', yaxis_title='Preço (US$)', width=1000)   
    st.plotly_chart(fig)

    st.write("**Resultados:**")
    st.write(f"**WMAPE**: {wmape4:.2%}")
    st.write(f"**RMSE**: {rmse4:.2f}")
    st.write(f"**MAPE**: {mape4:.2%}")

    paragrafo27_tab4 = f'<p style="text-align: justify;">Utilizando a sazonalidade de {best_season_length_seso} dias, o modelo Seasonal Exponential Smoothing Optimized conseguiu obter bons resultados (WMAPE: {wmape4:.2%}, RMSE: {rmse4:.2f} e MAPE: {mape4:.2%}), porém não conseguiu superar o modelo Seasonal Window Average.</p>'
    st.markdown(paragrafo27_tab4, unsafe_allow_html=True)


    st.markdown("##### AutoARIMA")

    paragrafo32_tab4 = '<p style="text-align: justify;">AutoARIMA é uma abordagem automatizada baseada em aprendizado de máquina para ajustar modelos ARIMA (AutoRegressive Integrated Moving Average). Ele procura automaticamente pelos melhores parâmetros do modelo ARIMA para a série temporal fornecida. Isso inclui termos autoregressivos (AR), termos de diferenciação (I) e termos de média móvel (MA). Para a execução do AutoARIMA, foram utilizados os dados da série convertida em estacionária.</p>'
    st.markdown(paragrafo32_tab4, unsafe_allow_html=True)   
    

    pipeline3_tab4_sf = Pipeline([
        ('transform_index_to_column', TransformIndexToColumn()),        
        ('rename_columns', RenameColumns()),        
        ('cast_to_datetime', CastToDatetime()),  
        ('fill_missing_data', FillMissingData()),
        ('add_column', AddColumn())
    ])

    df_pipe3_tab4_sf = pipeline3_tab4_sf.transform(df_preco_petroleo_s) 

    treino_auto_arima = df_pipe3_tab4_sf[(df_pipe3_tab4_sf['ds'] >= pd.to_datetime('2022-10-01')) & (df_pipe3_tab4_sf['ds'] < pd.to_datetime('2023-10-01'))]
    valid_auto_arima = df_pipe3_tab4_sf[df_pipe3_tab4_sf['ds'] >= pd.to_datetime('2023-10-01')]

    h = valid_auto_arima.index.nunique()

    model_a = StatsForecast(models=[AutoARIMA(season_length=30)], freq='D', n_jobs=-1)
    model_a.fit(treino_auto_arima)

    forecast_autorima = model_a.predict(h=h, level=[90])
    forecast_autorima = forecast_autorima.reset_index().merge(valid_auto_arima, on=['ds', 'unique_id'], how='left')

    wmape5 = wmape(forecast_autorima['y'].values, forecast_autorima['AutoARIMA'].values) 
    mape5 = mape(forecast_autorima['y'], forecast_autorima['AutoARIMA'])

    df_preco_petroleo_s_filtered = df_preco_petroleo_s[df_preco_petroleo_s['ds'] >= pd.to_datetime('2023-06-01')]
     
    fig = go.Figure()

    # série temporal prevista e intervalo de confiança
    fig.add_trace(go.Scatter(x=forecast_autorima['ds'], y=forecast_autorima['AutoARIMA'], mode='lines', name='Previsão (AutoARIMA)', line=dict(color='rgb(101, 101, 101)')))
    fig.add_trace(go.Scatter(x=forecast_autorima['ds'], y=forecast_autorima['AutoARIMA-lo-90'], fill=None, mode='lines', line=dict(color='rgb(228, 228, 228)', width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=forecast_autorima['ds'], y=forecast_autorima['AutoARIMA-hi-90'], fill='tonexty', mode='lines', line=dict(color='rgb(228, 228, 228)', width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=df_preco_petroleo_s_filtered['ds'], y=df_preco_petroleo_s_filtered['y'], mode='lines', name='Valor Real', line=dict(color='blue')))              
  
    fig.update_layout(title='Previsão do Preço (US$) do Petróleo Brent Utilizando o Modelo AutoARIMA', xaxis_title='Data', yaxis_title='Preço (US$)', width=1000)   
    st.plotly_chart(fig)

    st.write("**Resultados:**")
    st.write(f"**WMAPE**: {wmape5:.2%}")   
    st.write(f"**MAPE**: {mape5:.2%}")

    paragrafo33_tab4 = '<p style="text-align: justify;">Conforme visto, devido a natureza da série, o AutoARIMA acabou obtendo resultados péssimos.</p>'
    st.markdown(paragrafo33_tab4, unsafe_allow_html=True)
    
    
    st.markdown("### Conclusão")
    paragrafo34_tab4 = '<p style="text-align: justify;">Com base na análise comparativa dos modelos Naive, Seasonal Naive, Seasonal Window Average, Seasonal Exponential Smoothing Optimized e AutoARIMA, aplicados à série temporal de preços do petróleo Brent, obtidos do site do IPEA, e utilizando métricas de validação como WMAPE, MAPE e RMSE, constatou-se que o modelo que apresentou os resultados mais satisfatórios e que melhor capturou o comportamento dos dados foi o <b>Seasonal Window Average</b>.</p>'
    st.markdown(paragrafo34_tab4, unsafe_allow_html=True)

     
with tab5:
   
    st.markdown("## Detalhamento Técnico") 

    st.image('./imagens/arquitetura.png', caption='Arquitetura do projeto')
    texto = """
        A arquitetura do trabalho desenvolvido segue os seguintes passos:

        + **Web Scraping do Site do IPEA:** Os dados do IPEA, referente ao preço do petróleo bruto Brent, são extraídos regularmente. Esse processo é automatizado por meio do GitHub Actions, garantindo que o script seja executado diariamente às 10h da manhã.

        + **Armazenamento no Google BigQuery:** Os dados extraídos são armazenados na tabela "petroleo_brent" no Google BigQuery. Isso proporciona uma solução escalável e eficiente para armazenamento e gerenciamento de grandes conjuntos de dados.

        + **Notebook - Análise dos Modelos:** Um notebook foi desenvolvido com o propósito de realizar a avaliação de desempenho dos modelos Naive, Seasonal Naive, Seasonal Window Average, Seasonal Exponential Smoothing Optimized e AutoARIMA. Este notebook se conecta ao BigQuery para extrair os dados. Após a análise o modelo que obteve a melhor performance é extraído por meio do Joblib.

        + **Aplicação Streamlit:** Além do notebook, uma aplicação Streamlit foi desenvolvida para oferecer uma interface interativa para a análise das séries temporais. Essa aplicação também se conecta ao BigQuery para extrair dados em tempo real, fornecendo uma visualização dinâmica e amigável para os usuários finais.

        + **Dashboard Power BI:** Um dashboard foi desenvolvido no Power BI com o objetivo de oferecer insights relevantes sobre a variação do preço do petróleo. No entanto, devido ao fato de o Power BI ser uma aplicação paga para publicação de dashboards online, não foi possível disponibilizar a visualização de forma acessível pela web. Para contornar essa limitação, um arquivo .pbix foi gerado e está disponível para download. Dessa forma, os interessados podem baixar o arquivo e executar o dashboard localmente no Power BI para explorar e analisar as informações relacionadas à variação de preços do petróleo.
       
    """
    st.markdown(texto) 