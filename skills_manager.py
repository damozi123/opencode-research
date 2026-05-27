import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import httpx

REPOS_INDEX = Path(__file__).parent / "repos.json"

SKILL_DIRS = [
    Path.home() / ".config" / "opencode" / "skills",
    Path.cwd() / ".opencode" / "skills",
    Path.home() / ".claude" / "skills",
    Path.cwd() / ".claude" / "skills",
]


def _load_repos() -> dict:
    with open(REPOS_INDEX, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_install_dir(global_install: bool = False) -> Path:
    if global_install:
        d = Path.home() / ".config" / "opencode" / "skills"
    else:
        d = Path.cwd() / ".opencode" / "skills"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── 搜索 ─────────────────────────────────────────────────────────

def search_repos(query: str) -> str:
    """搜索 Skills 仓库，返回匹配的仓库列表"""
    data = _load_repos()
    q = query.lower()
    results = []
    for repo in data["repos"]:
        name = repo["name"].lower()
        desc = repo["description"].lower()
        tags = " ".join(repo.get("tags", [])).lower()
        if q in name or q in desc or q in tags:
            results.append(repo)
    if not results:
        return f"未找到与 '{query}' 相关的 Skills 仓库。"
    lines = []
    for r in results:
        lines.append(f"【{r['name']}】⭐ {r['stars']/1000:.1f}k | {r['skills']} 个 Skills")
        lines.append(f"  描述: {r['description']}")
        lines.append(f"  标签: {', '.join(r.get('tags', []))}")
        lines.append(f"  安装: {r['url']}")
        lines.append("")
    return "\n".join(lines)


def search_skills(query: str) -> str:
    """搜索所有已知仓库中的 Skills 内容，并下载可用 Skills 列表"""
    data = _load_repos()
    q = query.lower()
    results = []
    # Search repos first
    for repo in data["repos"]:
        name = repo["name"].lower()
        desc = repo["description"].lower()
        tags = " ".join(repo.get("tags", [])).lower()
        if q in name or q in desc or q in tags:
            results.append(repo)

    if not results:
        return f"未找到与 '{query}' 相关的 Skills。尝试用其他关键词搜索。"

    lines = [f"\"{query}\" 匹配 {len(results)} 个仓库：\n"]
    for r in results:
        lines.append(f"【{r['name']}】⭐ {r['stars']/1000:.1f}k")
        lines.append(f"  {r['description'][:100]}")
        lines.append(f"  安装命令: skills-manager install \"{r['name']}\"")
        lines.append("")
    return "\n".join(lines)


# ── 安装 ─────────────────────────────────────────────────────────

def _clone_repo(url: str, branch: str = "main") -> Path:
    tmp = tempfile.mkdtemp(prefix="skills_")
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", branch, url, tmp],
        capture_output=True, text=True, check=False
    )
    return Path(tmp)


async def _get_github_contents(owner: str, repo: str, path: str = "", branch: str = "main") -> list:
    """Fetch directory contents from GitHub API"""
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers={"Accept": "application/vnd.github.v3+json"})
        resp.raise_for_status()
        return resp.json()


async def install_skill(repo_name: str, skill_name: str | None = None, global_install: bool = False) -> str:
    """从 GitHub 仓库安装 Skill 到本地"""
    data = _load_repos()
    target_repo = None
    for r in data["repos"]:
        if repo_name.lower() in r["name"].lower():
            target_repo = r
            break

    if not target_repo:
        return f"未找到仓库 '{repo_name}'。请先用 search_repos 查找可用仓库。"

    url = target_repo["url"]
    branch = target_repo.get("branch", "main")
    install_dir = _get_install_dir(global_install)

    # Parse GitHub URL
    parts = url.rstrip("/").split("/")
    if "github.com" not in url:
        return "仅支持 GitHub 仓库。"
    
    try:
        gh_idx = parts.index("github.com")
        owner = parts[gh_idx + 1]
        repo = parts[gh_idx + 2]
    except (ValueError, IndexError):
        return f"无法解析 GitHub URL: {url}"

    # Try to get skill list from GitHub API
    try:
        skill_path = target_repo.get("install_path", "skills")
        contents = await _get_github_contents(owner, repo, skill_path, branch)
    except Exception:
        # Fallback: clone the repo
        try:
            tmpdir = _clone_repo(url, branch)
            skill_dir = tmpdir / skill_path
            if not skill_dir.exists():
                return f"仓库中未找到 skills 目录: {skill_path}"
            # List skills
            items = list(skill_dir.iterdir())
            lines = [f"仓库 '{target_repo['name']}' 中的 Skills:"]
            for item in sorted(items):
                if item.is_dir():
                    skill_md = item / "SKILL.md"
                    if skill_md.exists():
                        lines.append(f"  📁 {item.name}")
            lines.append(f"\n使用以下命令安装: skills-manager install \"{target_repo['name']}\" --skill <名称>")
            return "\n".join(lines)
        except Exception as e:
            return f"克隆失败: {e}"

    # List available skills
    if isinstance(contents, list):
        skill_dirs = [c for c in contents if c.get("type") == "dir"]
        if not skill_name:
            lines = [f"仓库 '{target_repo['name']}' 中的 Skills ({len(skill_dirs)} 个):"]
            for sd in skill_dirs:
                lines.append(f"  📁 {sd['name']}")
            lines.append(f"\n安装全部: skills-manager install \"{target_repo['name']}\" --all")
            lines.append(f"安装单个: skills-manager install \"{target_repo['name']}\" --skill <名称>")
            return "\n".join(lines)

        # Install specific skill
        if skill_name == "--all":
            targets = skill_dirs
        else:
            targets = [s for s in skill_dirs if s["name"] == skill_name]
            if not targets:
                return f"未找到 Skill '{skill_name}'。"

        installed = []
        for sd in targets:
            skill_contents = await _get_github_contents(owner, repo, f"{skill_path}/{sd['name']}", branch)
            dest_dir = install_dir / sd["name"]
            dest_dir.mkdir(parents=True, exist_ok=True)
            for item in skill_contents:
                if item["type"] == "file":
                    file_url = item["download_url"]
                    async with httpx.AsyncClient() as client:
                        file_resp = await client.get(file_url)
                        file_resp.raise_for_status()
                        dest_path = dest_dir / item["name"]
                        with open(dest_path, "w", encoding="utf-8") as f:
                            f.write(file_resp.text)
            installed.append(sd["name"])

        install_path = install_dir
        return f"已安装 {len(installed)} 个 Skill: {', '.join(installed)}\n安装位置: {install_path}"

    return f"无法解析仓库内容: {contents}"


# ── 管理 ─────────────────────────────────────────────────────────

def list_installed() -> str:
    """列出所有已安装的 Skills"""
    found = []
    for base_dir in SKILL_DIRS:
        if not base_dir.exists():
            continue
        for item in sorted(base_dir.iterdir()):
            if item.is_dir():
                skill_md = item / "SKILL.md"
                disabled_md = item / "SKILL.md.disabled"
                if skill_md.exists():
                    found.append((item, skill_md, base_dir, True))
                elif disabled_md.exists():
                    found.append((item, disabled_md, base_dir, False))

    if not found:
        return "没有已安装的 Skills。"

    lines = [f"已安装 {len(found)} 个 Skills:\n"]
    for item, md_file, base_dir, enabled in found:
        status = "启用" if enabled else "禁用"
        # Extract description from SKILL.md
        desc = ""
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("description:"):
                        desc = line.split(":", 1)[1].strip().strip('"').strip("'")
                        break
        except Exception:
            pass
        lines.append(f"  [{'o' if enabled else 'x'}] {item.name} ({status})")
        lines.append(f"       路径: {base_dir / item.name}")
        if desc:
            lines.append(f"       描述: {desc[:80]}")
        lines.append("")
    return "\n".join(lines)


def enable_skill(name: str, enable: bool = True) -> str:
    """启用或禁用一个 Skill"""
    for base_dir in SKILL_DIRS:
        skill_dir = base_dir / name
        if not skill_dir.exists():
            continue
        enabled_path = skill_dir / "SKILL.md"
        disabled_path = skill_dir / "SKILL.md.disabled"
        if enable and disabled_path.exists():
            disabled_path.rename(enabled_path)
            return f"已启用 Skill: {name}"
        elif not enable and enabled_path.exists():
            enabled_path.rename(disabled_path)
            return f"已禁用 Skill: {name}"
        else:
            status = "已启用" if enabled_path.exists() else "已禁用"
            return f"Skill '{name}' 已经是{status}状态。"
    return f"未找到 Skill '{name}'。"


def remove_skill(name: str) -> str:
    """删除一个已安装的 Skill"""
    for base_dir in SKILL_DIRS:
        skill_dir = base_dir / name
        if skill_dir.exists():
            shutil.rmtree(skill_dir)
            return f"已删除 Skill: {name} (来自 {base_dir / name})"
    return f"未找到 Skill '{name}'。"


def recommend_skills(context: str = "") -> str:
    """根据项目上下文智能推荐 Skills"""
    data = _load_repos()
    c = context.lower()
    results = []
    for repo in data["repos"]:
        tags = " ".join(repo.get("tags", [])).lower()
        desc = repo["description"].lower()
        name = repo["name"].lower()
        keywords = set()
        for tag in repo.get("tags", []):
            keywords.add(tag)
        score = 0
        for kw in keywords:
            if kw in c:
                score += 1
        if score > 0 or not context:
            results.append((score, repo))

    results.sort(key=lambda x: x[0], reverse=True)
    top = [r for s, r in results[:5]]

    if not context:
        lines = ["推荐所有可用 Skills 仓库: (按热度)"]
        top = sorted(data["repos"], key=lambda r: r["stars"], reverse=True)[:10]
    else:
        lines = [f"根据 '{context}' 推荐以下 Skills:"]
        if not top:
            lines.append("未找到匹配的 Skills。")
            return "\n".join(lines)

    for r in top:
        lines.append(f"  ★ {r['name']} | {r['stars']/1000:.1f}k | {r['skills']}个")
        lines.append(f"    {r['description'][:90]}")

    lines.append(f"\n安装: skills-manager install \"<仓库名>\"")
    return "\n".join(lines)


def smart_install(query: str, global_install: bool = True) -> str:
    """智能安装：搜索并自动安装匹配的 Skills"""
    data = _load_repos()
    q = query.lower()
    matches = []
    for repo in data["repos"]:
        tags = " ".join(repo.get("tags", [])).lower()
        desc = repo["description"].lower()
        name = repo["name"].lower()
        all_text = f"{name} {desc} {tags}"
        if q in all_text:
            matches.append(repo)

    if not matches:
        return search_repos(query)

    if len(matches) == 1:
        return f"找到一个匹配仓库:\n  {matches[0]['name']}\n  {matches[0]['url']}\n\n使用 sync install 或 skills-manager install 安装。"

    lines = [f"找到 {len(matches)} 个匹配仓库:"]
    for r in matches:
        lines.append(f"  {r['name']} - {r['url']}")
    lines.append(f"\n请选择具体仓库安装。")
    return "\n".join(lines)


# ── 导出给 MCP Server 使用 ───────────────────────────────────────

SKILL_MANAGER_FUNCTIONS = {
    "skill_search": search_repos,
    "skill_install": lambda repo_name, skill_name=None: install_skill(repo_name, skill_name),
    "skill_list": list_installed,
    "skill_enable": lambda name: enable_skill(name, True),
    "skill_disable": lambda name: enable_skill(name, False),
    "skill_remove": remove_skill,
    "skill_recommend": recommend_skills,
    "skill_smart_install": smart_install,
}


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Skills 管家")
        print("  search <关键词>     搜索可用 Skills 仓库")
        print("  install <仓库名>    安装 Skills")
        print("  list                列出已安装的 Skills")
        print("  enable <名称>       启用 Skill")
        print("  disable <名称>      禁用 Skill")
        print("  remove <名称>       删除 Skill")
        print("  recommend [上下文]   智能推荐 Skills")
        print("  smart <关键词>      智能搜索安装")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "search" and len(sys.argv) > 2:
        print(search_repos(sys.argv[2]))
    elif cmd == "find" and len(sys.argv) > 2:
        print(search_skills(sys.argv[2]))
    elif cmd == "install" and len(sys.argv) > 2:
        import asyncio
        repo_name = sys.argv[2]
        skill_name = None
        for a in sys.argv[3:]:
            if a.startswith("--skill="):
                skill_name = a.split("=", 1)[1]
        print(asyncio.run(install_skill(repo_name, skill_name)))
    elif cmd == "list":
        print(list_installed())
    elif cmd == "enable" and len(sys.argv) > 2:
        print(enable_skill(sys.argv[2], True))
    elif cmd == "disable" and len(sys.argv) > 2:
        print(enable_skill(sys.argv[2], False))
    elif cmd == "remove" and len(sys.argv) > 2:
        print(remove_skill(sys.argv[2]))
    elif cmd == "recommend":
        ctx = sys.argv[2] if len(sys.argv) > 2 else ""
        print(recommend_skills(ctx))
    elif cmd == "smart" and len(sys.argv) > 2:
        print(smart_install(sys.argv[2]))
    else:
        print("未知命令，运行无参数查看帮助。")
