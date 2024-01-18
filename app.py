import pandas as pd
import os
from google.cloud.bigquery.client import Client
import streamlit as st
import plotly.express as px
from utils import RenameColumns, CastToFloat, CastToDatetime, FillMissingData, AddColumn
from sklearn.pipeline import Pipeline
import joblib
from joblib import load
from datetime import datetime, timedelta
import plotly.graph_objects as go


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'fiap-tech-challenge-4-5cd1d93599ab.json'
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
    '''
    ## Dashboard

    '''

with tab3:
    '''
    ## Predição do Preço Petróleo Brent  
    '''

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
        data_especifica = datetime.strptime('2024-01-08', '%Y-%m-%d').date()

        diferenca_em_dias = (data_atual_tab3 - data_especifica).days
        qtde_dias_a_predizer = input_days_to_predict + diferenca_em_dias + 1


        model = joblib.load('modelo/sm.joblib')
        final_pred = model.predict(h = qtde_dias_a_predizer) 


        # transformações para melhorar a exibição dos dados na tabela
        final_pred_filtrado = final_pred[final_pred['ds'] > datetime.now()] 
        final_pred_filtrado.rename(columns= {'ds' : 'Data', 'SeasWA': 'Preço Predito (US$)'}, inplace=True)
        final_pred_filtrado['Data'] = final_pred_filtrado['Data'].dt.normalize()  
        final_pred_filtrado.reset_index(drop=True, inplace=True)

        st.dataframe(final_pred_filtrado)

       
        # transformações para que o gráfico exiba além do período predito, os dados dos 3 meses anteriores
        data_tres_meses_atras = data_atual_tab3 - timedelta(days=3 * 30)
        data_ha_3_meses = data_atual_tab3 - timedelta(days=3 * 30)

        df_pipe_tab3 = df_pipe_tab3[df_pipe_tab3['ds'] >= pd.to_datetime(data_ha_3_meses)]

        df_pipe_tab3.rename(columns= {'ds' : 'Data', 'y': 'Preço Predito (US$)'}, inplace=True)
        df_pipe_tab3_filtrado = df_pipe_tab3[['Data', 'Preço Predito (US$)']]   

        df_resultado = pd.concat([df_pipe_tab3_filtrado, final_pred_filtrado], ignore_index=True).sort_values(by='Data')        
        df_resultado.rename(columns= {'Preço Predito (US$)' : 'Preco_pretroleo_brent'}, inplace=True)


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
        data_posterior = data_atual_tab3 + timedelta(days= qtde_dias_a_predizer/2)
        
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
 
        st.plotly_chart(fig)




with tab4:
    '''
    ## Performance dos Modelos

    '''    

    

with tab5:
    '''
    ## Detalhamento Técnico

    '''