import json
import sqlite3
import subprocess
import tempfile
import os
import ast as ast_module
import html.parser
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

import skills_manager

server = Server("academic-mcp")
http_client = httpx.AsyncClient(timeout=30, follow_redirects=True)

# ── Tool definitions ─────────────────────────────────────────────

TOOLS = [
    Tool(name="db_query", description="Execute a SQL query on a SQLite database file",
         inputSchema={"type": "object", "properties": {
             "db_path": {"type": "string", "description": "Absolute path to the SQLite database file"},
             "sql": {"type": "string", "description": "SQL query to execute"},
             "params": {"type": "array", "description": "Optional parameters for parameterized queries", "items": {"type": "string"}}
         }, "required": ["db_path", "sql"]}),
    Tool(name="db_list_tables", description="List all tables and their schemas in a SQLite database file",
         inputSchema={"type": "object", "properties": {
             "db_path": {"type": "string", "description": "Absolute path to the SQLite database file"}
         }, "required": ["db_path"]}),
    Tool(name="web_fetch", description="Fetch content from a URL and return it as plain text",
         inputSchema={"type": "object", "properties": {
             "url": {"type": "string", "description": "URL to fetch"}
         }, "required": ["url"]}),
    Tool(name="web_search", description="Search the web using DuckDuckGo",
         inputSchema={"type": "object", "properties": {
             "query": {"type": "string", "description": "Search query"},
             "max_results": {"type": "integer", "description": "Maximum number of results (default 10)"}
         }, "required": ["query"]}),
    Tool(name="search_arxiv", description="Search arXiv for academic papers",
         inputSchema={"type": "object", "properties": {
             "query": {"type": "string", "description": "Search query"},
             "max_results": {"type": "integer", "description": "Maximum results (default 10)"},
             "category": {"type": "string", "description": "Category filter (e.g. cs.AI, physics)"}
         }, "required": ["query"]}),
    Tool(name="search_pubmed", description="Search PubMed for biomedical papers",
         inputSchema={"type": "object", "properties": {
             "query": {"type": "string", "description": "Search query"},
             "max_results": {"type": "integer", "description": "Maximum results (default 10)"}
         }, "required": ["query"]}),
    Tool(name="search_semantic_scholar", description="Search Semantic Scholar for papers",
         inputSchema={"type": "object", "properties": {
             "query": {"type": "string", "description": "Search query"},
             "max_results": {"type": "integer", "description": "Maximum results (default 10)"}
         }, "required": ["query"]}),
    Tool(name="python_run", description="Execute Python code and return the output",
         inputSchema={"type": "object", "properties": {
             "code": {"type": "string", "description": "Python source code to execute"}
         }, "required": ["code"]}),
    Tool(name="python_analyze", description="Analyze structure of a Python file (imports, functions, classes)",
         inputSchema={"type": "object", "properties": {
             "file_path": {"type": "string", "description": "Absolute path to the Python file"}
         }, "required": ["file_path"]}),
    Tool(name="python_modify", description="Rewrite a Python file with new content",
         inputSchema={"type": "object", "properties": {
             "file_path": {"type": "string", "description": "Absolute path to the Python file"},
             "file_content": {"type": "string", "description": "New source code for the file"}
         }, "required": ["file_path", "file_content"]}),
    # Skills 管家
    Tool(name="skill_search", description="Search available skill repositories by keyword",
         inputSchema={"type": "object", "properties": {
             "query": {"type": "string", "description": "Search keyword (e.g. 科研, 论文, Python, 数据库)"}
         }, "required": ["query"]}),
    Tool(name="skill_list", description="List all installed skills and their status",
         inputSchema={"type": "object", "properties": {}}),
    Tool(name="skill_install", description="Install skills from a GitHub repository. First use skill_search to find repos.",
         inputSchema={"type": "object", "properties": {
             "repo_name": {"type": "string", "description": "Repository name from search results"},
             "skill_name": {"type": "string", "description": "Specific skill name to install (optional, installs all if omitted)"}
         }, "required": ["repo_name"]}),
    Tool(name="skill_enable", description="Enable an installed skill",
         inputSchema={"type": "object", "properties": {
             "name": {"type": "string", "description": "Skill name to enable"}
         }, "required": ["name"]}),
    Tool(name="skill_disable", description="Disable an installed skill",
         inputSchema={"type": "object", "properties": {
             "name": {"type": "string", "description": "Skill name to disable"}
         }, "required": ["name"]}),
    Tool(name="skill_remove", description="Remove an installed skill",
         inputSchema={"type": "object", "properties": {
             "name": {"type": "string", "description": "Skill name to remove"}
         }, "required": ["name"]}),
    Tool(name="skill_recommend", description="Recommend skills based on project context",
         inputSchema={"type": "object", "properties": {
             "context": {"type": "string", "description": "Project context or keywords (optional)"}
         }}),
    Tool(name="skill_smart_install", description="Smart search and install: find the best matching skills repo for your needs",
         inputSchema={"type": "object", "properties": {
             "query": {"type": "string", "description": "What kind of skills do you need? (e.g. 科研论文, 网页抓取, Python开发)"}
         }, "required": ["query"]}),
    Tool(name="skill_scene", description="Switch skill scene/profile - batch enable/disable skills for a specific workflow. Also use 'list' or 'presets' to see available scenes.",
         inputSchema={"type": "object", "properties": {
             "scene": {"type": "string", "description": "Scene name: 科研写作, 地学分析, 办公文档, 数据分析, 全栈开发, 全开, 全关, or list to show all"}
         }, "required": ["scene"]}),
]

# ── Handlers ─────────────────────────────────────────────────────


@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    return TOOLS


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    args = {k: v for k, v in arguments.items() if v is not None}

    if name == "db_query":
        return await _db_query(args.get("db_path", ""), args.get("sql", ""), args.get("params", []))
    elif name == "db_list_tables":
        return await _db_list_tables(args.get("db_path", ""))
    elif name == "web_fetch":
        return await _web_fetch(args.get("url", ""))
    elif name == "web_search":
        return await _web_search(args.get("query", ""), args.get("max_results", 10))
    elif name == "search_arxiv":
        return await _search_arxiv(args.get("query", ""), args.get("max_results", 10), args.get("category", ""))
    elif name == "search_pubmed":
        return await _search_pubmed(args.get("query", ""), args.get("max_results", 10))
    elif name == "search_semantic_scholar":
        return await _search_semantic_scholar(args.get("query", ""), args.get("max_results", 10))
    elif name == "python_run":
        return await _python_run(args.get("code", ""))
    elif name == "python_analyze":
        return await _python_analyze(args.get("file_path", ""))
    elif name == "python_modify":
        return await _python_modify(args.get("file_path", ""), args.get("file_content", ""))
    elif name.startswith("skill_"):
        return await _skill_handler(name, args)
    else:
        return [TextContent(type="text", text=f"未知工具: {name}")]


# ── Skill 工具调度 ───────────────────────────────────────────────


async def _skill_handler(name: str, args: dict) -> list[TextContent]:
    if name == "skill_search":
        return [TextContent(type="text", text=skills_manager.search_repos(args.get("query", "")))]
    elif name == "skill_list":
        return [TextContent(type="text", text=skills_manager.list_installed())]
    elif name == "skill_install":
        return [TextContent(type="text", text=await skills_manager.install_skill(
            args.get("repo_name", ""), args.get("skill_name")))]
    elif name == "skill_enable":
        return [TextContent(type="text", text=skills_manager.enable_skill(args.get("name", ""), True))]
    elif name == "skill_disable":
        return [TextContent(type="text", text=skills_manager.enable_skill(args.get("name", ""), False))]
    elif name == "skill_remove":
        return [TextContent(type="text", text=skills_manager.remove_skill(args.get("name", "")))]
    elif name == "skill_recommend":
        return [TextContent(type="text", text=skills_manager.recommend_skills(args.get("context", "")))]
    elif name == "skill_smart_install":
        return [TextContent(type="text", text=skills_manager.smart_install(args.get("query", "")))]
    elif name == "skill_scene":
        return [TextContent(type="text", text=_scene_switch(args.get("scene", "")))]
    return [TextContent(type="text", text=f"未知 Skill 工具: {name}")]


# ── 场景切换 ──────────────────────────────────────────────────────

SCENES: dict[str, tuple[str, ...]] = {
    "list": (),
    "presets": (),

    "科研写作": (
        "academic-writing", "literature-review", "peer-review",
        "pdf", "docx", "matplotlib", "exploratory-data-analysis",
    ),
    "地学分析": (
        "geomaster", "matplotlib", "exploratory-data-analysis",
    ),
    "办公文档": (
        "docx", "pptx", "xlsx", "pdf", "infographics",
    ),
    "数据分析": (
        "exploratory-data-analysis", "matplotlib", "infographics",
    ),
    "全栈开发": (
        "frontend-design", "web-artifacts-builder", "webapp-testing",
        "mcp-builder", "skill-creator", "canvas-design",
    ),
    "产品设计": (
        "frontend-design", "canvas-design", "algorithmic-art",
        "web-artifacts-builder", "infographics",
    ),

    "全开": (),
    "全关": (),
}

SKILL_LOCATIONS = [
    Path.home() / ".opencode" / "skills",
    Path.home() / ".claude" / "skills",
    Path.home() / ".config" / "opencode" / "skills",
]


def _all_skill_dirs() -> list[Path]:
    """遍历所有技能目录，找出全部已安装的技能"""
    found = set()
    for loc in SKILL_LOCATIONS:
        if not loc.exists():
            continue
        for item in loc.iterdir():
            if item.is_dir():
                sk = item / "SKILL.md"
                ds = item / "SKILL.md.disabled"
                if sk.exists() or ds.exists():
                    found.add(item.name)
    return sorted(found)


def _scene_switch(scene: str) -> str:
    if scene in ("list", "presets"):
        lines = ["可用场景预设:"]
        for name, skills in SCENES.items():
            if name in ("list", "presets"):
                continue
            if name in ("全开", "全关"):
                lines.append(f"  📌 {name} — 全部Skills{'启用' if name == '全开' else '禁用'}")
            else:
                lines.append(f"  📌 {name} — {', '.join(skills)}")
        lines.append(f"\n使用: skill_scene scene=\"场景名\"")
        return "\n".join(lines)

    if scene not in SCENES:
        return f"未知场景 '{scene}'。用 skill_scene scene=\"list\" 查看可用场景。"

    all_skills = _all_skill_dirs()
    if not all_skills:
        return "未找到任何已安装的 Skills。"

    # 全开/全关
    if scene == "全开":
        count = 0
        for name in all_skills:
            for loc in SKILL_LOCATIONS:
                d = loc / name / "SKILL.md.disabled"
                if d.exists():
                    d.rename(loc / name / "SKILL.md")
                    count += 1
                    break
        return f"已启用全部 {count} 个 Skills。"

    if scene == "全关":
        count = 0
        for name in all_skills:
            for loc in SKILL_LOCATIONS:
                d = loc / name / "SKILL.md"
                if d.exists():
                    d.rename(loc / name / "SKILL.md.disabled")
                    count += 1
                    break
        return f"已禁用全部 {count} 个 Skills。"

    # 场景切换
    scene_skills = set(SCENES[scene])
    enabled_list = []
    disabled_list = []

    for name in all_skills:
        should_enable = name in scene_skills
        for loc in SKILL_LOCATIONS:
            skill_dir = loc / name
            if should_enable:
                ds = skill_dir / "SKILL.md.disabled"
                if ds.exists():
                    ds.rename(skill_dir / "SKILL.md")
                    enabled_list.append(name)
                    break
            else:
                sk = skill_dir / "SKILL.md"
                if sk.exists():
                    sk.rename(skill_dir / "SKILL.md.disabled")
                    disabled_list.append(name)
                    break

    lines = [f"已切换到场景: {scene}"]
    if enabled_list:
        lines.append(f"  启用: {', '.join(sorted(enabled_list))}")
    if disabled_list:
        lines.append(f"  禁用: {', '.join(sorted(disabled_list))}")
    return "\n".join(lines)


# ── Database ─────────────────────────────────────────────────────


async def _db_query(db_path: str, sql: str, params: list | None = None) -> list[TextContent]:
    if params is None:
        params = []
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(sql, params)
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        return [TextContent(type="text", text="查询成功，但没有返回结果。")]
    columns = rows[0].keys()
    lines = [" | ".join(columns), "-" * 40]
    for row in rows:
        lines.append(" | ".join(str(v) for v in row))
    return [TextContent(type="text", text="\n".join(lines))]


async def _db_list_tables(db_path: str) -> list[TextContent]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = cursor.fetchall()
    result = []
    for (table_name,) in tables:
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        schema = cursor.fetchone()
        result.append(f"表: {table_name}")
        if schema and schema[0]:
            result.append(schema[0])
        result.append("")
    conn.close()
    if not result:
        return [TextContent(type="text", text="数据库中没有找到任何表。")]
    return [TextContent(type="text", text="\n".join(result))]


# ── Web ──────────────────────────────────────────────────────────


async def _web_fetch(url: str) -> list[TextContent]:
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"
    resp = await http_client.get(url)
    resp.raise_for_status()
    text = resp.text[:8000]
    return [TextContent(type="text", text=text)]


async def _web_search(query: str, max_results: int = 10) -> list[TextContent]:
    ddg_url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
    resp = await http_client.get(ddg_url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})

    class ResultParser(html.parser.HTMLParser):
        def __init__(self):
            super().__init__()
            self.results = []
            self.capture = False
            self.capture_link = False
            self.current = {}
            self.buffer = ""

        def handle_starttag(self, tag, attrs):
            d = dict(attrs)
            cls = d.get("class", "")
            if tag == "a" and "result__a" in cls:
                self.capture = True
                self.current = {"title": "", "url": "", "snippet": ""}
                href = d.get("href", "")
                if href.startswith("//"):
                    href = "https:" + href
                self.current["url"] = href
            if self.capture and tag == "a":
                self.capture_link = True
                self.buffer = ""
            if tag == "a" and "result__snippet" in cls:
                self.capture_link = True
                self.buffer = ""

        def handle_endtag(self, tag):
            if self.capture and tag == "a":
                if self.capture_link:
                    text = self.buffer.strip()
                    if self.current["title"]:
                        self.current["snippet"] = text
                    else:
                        self.current["title"] = text
                    self.buffer = ""
                    self.capture_link = False
            if tag == "div" and self.current.get("url"):
                self.results.append(self.current.copy())
                self.current = {}
                self.capture = False

        def handle_data(self, data):
            if self.capture_link:
                self.buffer += data

    parser = ResultParser()
    parser.feed(resp.text)

    lines = []
    for i, r in enumerate(parser.results[:max_results], 1):
        lines.append(f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet']}\n")
    if not lines:
        return [TextContent(type="text", text="没有找到搜索结果。")]
    return [TextContent(type="text", text="\n".join(lines))]


# ── Academic ─────────────────────────────────────────────────────


async def _search_arxiv(query: str, max_results: int = 10, category: str = "") -> list[TextContent]:
    search_query = f"all:{quote(query)}"
    if category:
        search_query = f"cat:{category}+AND+all:{quote(query)}"
    api_url = f"http://export.arxiv.org/api/query?search_query={search_query}&start=0&max_results={max_results}&sortBy=relevance&sortOrder=descending"
    resp = await http_client.get(api_url)
    ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    root = ET.fromstring(resp.text)
    entries = root.findall("atom:entry", ns)
    lines = []
    for entry in entries[:max_results]:
        title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
        url = entry.find("atom:id", ns).text.strip()
        summary = entry.find("atom:summary", ns).text.strip().replace("\n", " ")[:300]
        authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)]
        published = entry.find("atom:published", ns).text[:10]
        categories = [c.get("term") for c in entry.findall("arxiv:primary_category", ns)]
        lines.append(f"标题: {title}")
        lines.append(f"作者: {', '.join(authors[:5])}")
        lines.append(f"日期: {published}  类别: {', '.join(categories)}")
        lines.append(f"链接: {url}")
        lines.append(f"摘要: {summary}")
        lines.append("")
    if not lines:
        return [TextContent(type="text", text="没有找到相关论文。")]
    return [TextContent(type="text", text="\n".join(lines))]


async def _search_pubmed(query: str, max_results: int = 10) -> list[TextContent]:
    search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmax={max_results}&retmode=json&term={quote(query)}"
    resp = await http_client.get(search_url)
    data = resp.json()
    id_list = data.get("esearchresult", {}).get("idlist", [])
    if not id_list:
        return [TextContent(type="text", text="没有找到相关论文。")]
    fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&retmode=json&id={','.join(id_list)}"
    resp2 = await http_client.get(fetch_url)
    summary = resp2.json()
    lines = []
    for pmid in id_list:
        doc = summary.get("result", {}).get(pmid, {})
        title = doc.get("title", "N/A")
        authors = ", ".join(a["name"] for a in doc.get("authors", [])[:5])
        pubdate = doc.get("pubdate", "N/A")
        source = doc.get("source", "N/A")
        doi = ""
        for aid in doc.get("articleids", []):
            if aid.get("idtype") == "doi":
                doi = f"https://doi.org/{aid['value']}"
        lines.append(f"标题: {title}")
        lines.append(f"作者: {authors}")
        lines.append(f"期刊: {source} ({pubdate})")
        lines.append(f"PMID: {pmid}  DOI: {doi}")
        lines.append("")
    return [TextContent(type="text", text="\n".join(lines))]


async def _search_semantic_scholar(query: str, max_results: int = 10) -> list[TextContent]:
    api_url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={quote(query)}&limit={max_results}&fields=title,authors,year,externalIds,abstract,citationCount,url,venue"
    resp = await http_client.get(api_url)
    data = resp.json()
    lines = []
    for paper in data.get("data", []):
        title = paper.get("title", "N/A")
        authors = ", ".join(a["name"] for a in paper.get("authors", [])[:5])
        year = paper.get("year", "N/A")
        venue = paper.get("venue", "")
        venue_name = venue if isinstance(venue, str) else venue.get("name", "")
        abstract = (paper.get("abstract") or "").replace("\n", " ")[:300]
        citations = paper.get("citationCount", 0)
        url = paper.get("url", "")
        doi = paper.get("externalIds", {}).get("DOI", "")
        lines.append(f"标题: {title}")
        lines.append(f"作者: {authors}")
        lines.append(f"年份: {year}  引用: {citations}  期刊: {venue_name}")
        if doi:
            lines.append(f"DOI: https://doi.org/{doi}")
        if url:
            lines.append(f"链接: {url}")
        lines.append(f"摘要: {abstract}")
        lines.append("")
    if not lines:
        return [TextContent(type="text", text="没有找到相关论文。")]
    return [TextContent(type="text", text="\n".join(lines))]


# ── Python ───────────────────────────────────────────────────────


async def _python_run(code: str) -> list[TextContent]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        tmp_path = f.name
    try:
        result = subprocess.run(
            ["python", tmp_path], capture_output=True, text=True, timeout=30,
            cwd=tempfile.gettempdir())
        output = (result.stdout or "")
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return [TextContent(type="text", text=output[:4000])]
    except subprocess.TimeoutExpired:
        return [TextContent(type="text", text="代码执行超时（30秒）。")]
    finally:
        os.unlink(tmp_path)


async def _python_analyze(file_path: str) -> list[TextContent]:
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast_module.parse(source)
    lines = [f"文件: {file_path}", f"总行数: {len(source.splitlines())}", ""]
    imports, functions, classes = [], [], []
    for node in ast_module.walk(tree):
        if isinstance(node, ast_module.Import):
            for alias in node.names:
                imports.append(f"import {alias.name}")
        elif isinstance(node, ast_module.ImportFrom):
            names = ", ".join(a.name for a in node.names)
            imports.append(f"from {node.module} import {names}")
        elif isinstance(node, ast_module.FunctionDef):
            functions.append(f"def {node.name}(...)  行 {node.lineno}")
        elif isinstance(node, ast_module.ClassDef):
            classes.append(f"class {node.name}  行 {node.lineno}")
    if imports:
        lines.append("--- 导入 ---")
        for i in imports:
            lines.append(f"  {i}")
        lines.append("")
    if classes:
        lines.append("--- 类 ---")
        for c in classes:
            lines.append(f"  {c}")
        lines.append("")
    if functions:
        lines.append("--- 函数 ---")
        for f in functions:
            lines.append(f"  {f}")
        lines.append("")
    return [TextContent(type="text", text="\n".join(lines))]


async def _python_modify(file_path: str, file_content: str) -> list[TextContent]:
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(file_content)
    try:
        ast_module.parse(file_content)
        return [TextContent(type="text", text=f"文件已写入并通过语法检查: {file_path}")]
    except SyntaxError as e:
        return [TextContent(type="text", text=f"文件已写入但存在语法错误: {e}")]


# ── Entry ────────────────────────────────────────────────────────

async def main():
    async with stdio_server() as (reader, writer):
        await server.run(reader, writer, server.create_initialization_options())

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
