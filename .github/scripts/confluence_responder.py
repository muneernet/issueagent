# confluence_responder.py
import os, re, json, requests
from github import Github
import openai
from math import sqrt

openai.api_key = os.environ.get('OPENAI_API_KEY')
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')
CONFLUENCE_BASE = os.environ.get('CONFLUENCE_BASE').rstrip('/')
CONFLUENCE_USER = os.environ.get('CONFLUENCE_USER')
CONFLUENCE_TOKEN = os.environ.get('CONFLUENCE_TOKEN')

def confluence_search(q, limit=5):
    cql = f'text ~ "{q}"'
    url = f"{CONFLUENCE_BASE}/rest/api/content/search"
    params = {'cql': cql, 'limit': limit, 'expand': 'body.storage,version'}
    r = requests.get(url, params=params, auth=(CONFLUENCE_USER, CONFLUENCE_TOKEN))
    r.raise_for_status()
    return r.json().get('results', [])

def strip_html(html):
    return re.sub('<[^<]+?>', ' ', html).replace('\n',' ').strip()

def embed_text(text):
    r = openai.Embeddings.create(model='text-embedding-3-small', input=text)
    return r['data'][0]['embedding']

def cosine(a,b):
    dot=sum(x*y for x,y in zip(a,b))
    na=sum(x*x for x in a)
    nb=sum(y*y for y in b)
    return dot/(sqrt(na)*sqrt(nb)+1e-12)

def get_issue_context():
    # GitHub Actions provides payload at GITHUB_EVENT_PATH
    event_path = os.environ.get('GITHUB_EVENT_PATH')
    with open(event_path) as f:
        payload = json.load(f)
    issue = payload['issue']
    repo_full = payload['repository']['full_name']
    return issue, repo_full

def generate_reply(issue, top_pages):
    # simple prompt
    docs = "\n\n".join([f"Title: {p['page']['title']}\n{strip_html(p['page']['body']['storage']['value'])[:1000]}" for p in top_pages])
    prompt = f"""You are a helpful assistant. The user opened this issue:
Title: {issue['title']}
Body: {issue.get('body','')}

From Confluence docs:
{docs}

Write a concise reply to the issue referencing the docs and suggesting next steps. Keep it short (<= 300 words).
"""
    r = openai.ChatCompletion.create(model='gpt-4o-mini', messages=[{"role":"user","content":prompt}], max_tokens=400)
    return r['choices'][0]['message']['content'].strip()

def main():
    issue, repo_full = get_issue_context()
    query = issue['title'] or issue.get('body','')[:120]
    pages = confluence_search(query, limit=5)
    if not pages:
        comment = "I couldn't find matching Confluence docs. Could you add more details?"
    else:
        # embed and rank
        try:
            issue_emb = embed_text(issue['title'] + '\n' + issue.get('body', ''))
            scored = []
            for p in pages:
                text = strip_html(p['body']['storage']['value'])
                p_emb = embed_text(text[:3000])
                score = cosine(issue_emb, p_emb)
                scored.append({'page': p, 'score': score})
            scored.sort(key=lambda x: x['score'], reverse=True)
            top = scored[:2]
            comment = generate_reply(issue, top)
        except Exception as e:
            # fallback
            top_titles = '\n'.join(f"- {p['title']}" for p in pages[:2])
            comment = f"Couldn't run semantic search, but these pages may help:\n{top_titles}"

    # post comment using PyGithub
    g = Github(GITHUB_TOKEN)
    repo = g.get_repo(repo_full)
    repo.create_issue_comment(issue['number'], comment)
    print("Posted comment.")

if __name__ == "__main__":
    main()

