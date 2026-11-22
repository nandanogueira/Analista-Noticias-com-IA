# Analista de Notícias com IA

Aplicação que recebe uma pergunta em linguagem natural sobre qualquer assunto, usa um **modelo de linguagem (LLM)** para interpretar a intenção e **escolher a ferramenta adequada**, consulta a **NewsAPI em tempo real** e devolve um **resumo organizado dos principais acontecimentos com as fontes**.

## Como funciona

1. O usuário faz uma pergunta (ex.: *"O que aconteceu com a economia brasileira nesta semana?"*).
2. A aplicação valida a pergunta e a envia ao LLM junto com a descrição de 3 ferramentas.
3. O LLM interpreta a intenção e **escolhe a ferramenta** e os argumentos (palavras-chave, idioma, período, país...).
4. A aplicação executa a consulta **real** na NewsAPI usando `requests`.
5. O retorno JSON é **processado**: artigos removidos e duplicados são descartados, textos são limpos e os artigos são numerados.
6. O LLM recebe as notícias e redige: **Resumo → Principais acontecimentos → Contexto e análise → Observações**, citando cada notícia como `[n]`.
7. A interface mostra a resposta com citações clicáveis, os **cartões das fontes** e um painel explicando **qual ferramenta a IA escolheu e por quê**.

O diagrama completo está em [docs/diagrama.pdf](docs/diagrama.pdf).

## Ferramentas disponíveis para a IA

| Ferramenta | Endpoint NewsAPI | Quando a IA usa |
|---|---|---|
| `buscar_noticias` | `/v2/everything` | Assunto específico (tema, pessoa, empresa, evento) |
| `manchetes_principais` | `/v2/top-headlines` | "O que está acontecendo hoje?", destaques por categoria |
| `listar_fontes` | `/v2/top-headlines/sources` | "Quais veículos estão disponíveis?" |

## Tecnologias

- **Python 3.10+**
- **Requests**: consumo da NewsAPI e do LLM (sem SDKs proprietários)
- **NewsAPI**: busca de notícias
- **Groq** (IA principal) e **Google Gemini** (IA reserva): LLMs gratuitos acessados via `requests`, pelo endpoint compatível com OpenAI, com function calling. Se o Groq falhar (limite de cota, instabilidade, timeout), a pergunta é refeita automaticamente com o Gemini
- **Streamlit**: interface

## Como executar

```bash
# 1. Criar ambiente virtual (opcional, recomendado)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Configurar as chaves
copy .env.example .env          # Windows  (Linux/Mac: cp .env.example .env)
# edite o .env e preencha NEWSAPI_KEY, GROQ_API_KEY e GEMINI_API_KEY

# 4. Rodar
streamlit run app.py
```

A aplicação abre em http://localhost:8501. Todas as configurações ficam no `.env`; **ao editá-lo, reinicie o `streamlit run`**.

### Chaves necessárias (todas gratuitas)

| Serviço | Onde obter | Variável |
|---|---|---|
| NewsAPI | https://newsapi.org/register | `NEWSAPI_KEY` |
| Groq (principal) | https://console.groq.com/keys | `GROQ_API_KEY` |
| Google Gemini (reserva) | https://aistudio.google.com/apikey | `GEMINI_API_KEY` |

### Principal + reserva (fallback automático)

- A IA **principal** é o Groq (modelo padrão `openai/gpt-oss-120b`); a **reserva** é o Gemini (`gemini-3-flash-preview`).
- Se o Groq falhar por **limite de cota (429)**, erro 5xx, timeout, falta de conexão ou modelo indisponível, a aplicação **refaz a pergunta inteira** com o Gemini. A resposta mostra um aviso de que a reserva foi usada e o motivo da falha.
- Se só uma das chaves estiver configurada, só ela é usada. Se as duas falharem, o app mostra o motivo de cada uma.
- Chave inválida (401) também aciona a reserva, e o aviso informa o motivo.
- Para mudar os modelos, defina `GROQ_MODEL` / `GEMINI_MODEL` no `.env` (precisam suportar *function calling*). Se der erro 404, o modelo foi renomeado ou descontinuado: veja a lista atual em https://console.groq.com/docs/models e https://ai.google.dev/gemini-api/docs/models.
- A ordem é fixa (Groq → Gemini). Os limites gratuitos mudam com frequência; confira os seus nos consoles dos provedores.

## Testes

```bash
pip install pytest
python -m pytest -q
```

Os testes cobrem as partes determinísticas: validação, processamento, citações e tratamento de erros das ferramentas. A aplicação em si sempre faz consultas reais.

## Estrutura

```
├── app.py                 # Interface Streamlit
├── src/
│   ├── agent.py           # Orquestração LLM ↔ ferramentas
│   ├── tools.py           # Definição, validação e execução das ferramentas
│   ├── news_api.py        # Cliente NewsAPI (requests)
│   ├── llm_client.py      # Cliente LLM compatível com OpenAI (requests)
│   ├── processing.py      # Processamento do JSON da NewsAPI
│   ├── formatting.py      # Citações [n] → links
│   ├── config.py          # Provedores, limites e leitura de chaves
│   └── errors.py          # Exceções com mensagens amigáveis
├── docs/diagrama.pdf      # Diagrama único (arquitetura, fluxo, erros)
├── tests/test_core.py     # Testes unitários
├── requirements.txt
└── .env.example
```
