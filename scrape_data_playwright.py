import asyncio
import random
from turtle import title
import pandas as pd
from urllib.parse import urljoin
from playwright.async_api import async_playwright

BASE = "https://web-scraping.dev/"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

def polite_sleep():
    return asyncio.sleep(random.uniform(0.7, 1.5))

async def find_next_page(page):
    """
    Returns next page URL if exists, otherwise None.
    """
    # First try rel="next"
    nxt = await page.query_selector('a[rel="next"]')
    if nxt:
        href = await nxt.get_attribute("href")
        if href:
            return urljoin(BASE, href)

    # Try a link that contains "next"
    nxt2 = await page.query_selector('a:has-text("Next")')
    if nxt2:
        href = await nxt2.get_attribute("href")
        if href:
            return urljoin(BASE, href)

    return None


async def scrape_products(page):
    url = urljoin(BASE, "products?page=1")
    all_rows = []

    while url:
        print("📦 Products page:", url)
        await page.goto(url, wait_until="networkidle")

        # ✅ ждём пока появятся товары
        await page.wait_for_selector("h3.mb-0", timeout=10000)

        # ✅ карточки товара
        cards = await page.query_selector_all(".product, article, .card")

        for c in cards:
            title_el = await c.query_selector("h3.mb-0")
            desc_el = await c.query_selector("div.short-description")
            price_el = await c.query_selector("div.price")

            title = (await title_el.inner_text()).strip() if title_el else None
            short_description = (await desc_el.inner_text()).strip() if desc_el else None
            price = (await price_el.inner_text()).strip() if price_el else None

            if title:
                all_rows.append({
                    "title": title,
                    "short_description": short_description,
                    "price": price
                })

        # ✅ ПАГИНАЦИЯ: берем последнюю ссылку в paging (обычно это ">")
        paging_links = await page.query_selector_all("div.paging a")
        if paging_links:
            last_link = paging_links[-1]
            href = await last_link.get_attribute("href")

            # если href есть и это не текущая страница → идём дальше
            if href and href != url:
                url = urljoin(BASE, href)
            else:
                url = None
        else:
            url = None

        await polite_sleep()

    return pd.DataFrame(all_rows)



async def scrape_testimonials(page, max_scrolls=60):
    url = urljoin(BASE, "testimonials")
    all_rows = []

    print("💬 Testimonials page:", url)
    await page.goto(url, wait_until="networkidle")

    # ✅ ждём, пока появится первый testimonial
    await page.wait_for_selector("p.text", timeout=10000)

    prev_count = 0
    stable_rounds = 0

    # ✅ Infinite Scroll: скроллим вниз пока появляются новые отзывы
    for i in range(max_scrolls):
        current_count = await page.locator("p.text").count()
        print(f"🔽 Scroll {i+1}/{max_scrolls} | testimonials loaded: {current_count}")

        if current_count == prev_count:
            stable_rounds += 1
        else:
            stable_rounds = 0

        # ✅ если уже 3 скролла подряд ничего нового — стоп
        if stable_rounds >= 3:
            print("✅ No new testimonials after scrolling. Stopping.")
            break

        prev_count = current_count

        # ✅ Скроллим вниз
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")

        # ✅ Если есть спиннер — ждём пока исчезнет
        spinner = page.locator("#testimonials-spinner")
        if await spinner.count() > 0 and await spinner.is_visible():
            try:
                await spinner.wait_for(state="hidden", timeout=10000)
            except:
                pass

        await polite_sleep()

    # ✅ Сбор данных: текст + stars
    testimonials = await page.query_selector_all("div.testimonial, article, .card")

    for t in testimonials:
        text_el = await t.query_selector("p.text")
        rating_el = await t.query_selector("span.rating")

        text = (await text_el.inner_text()).strip() if text_el else None

        stars = 0
        if rating_el:
            svgs = await rating_el.query_selector_all("svg")
            stars = len(svgs)

        if text:
            all_rows.append({"stars": stars, "text": text})

    return pd.DataFrame(all_rows)



async def scrape_reviews(page):
    url = urljoin(BASE, "reviews")
    all_rows = []

    print("⭐ Reviews page:", url)
    await page.goto(url, wait_until="networkidle")

    # ✅ Ждем, пока появятся первые отзывы
    await page.wait_for_selector("div[data-testid='review']", timeout=10000)

    # ✅ Нажимаем Load More пока он есть
    while True:
        load_more = await page.query_selector("#page-load-more")
        if not load_more:
            print("✅ Load More button not found — finished loading.")
            break

        # иногда кнопка существует, но уже disabled/hidden
        is_visible = await load_more.is_visible()
        if not is_visible:
            print("✅ Load More not visible — finished loading.")
            break

        # ✅ запоминаем сколько отзывов было ДО клика
        before_count = await page.locator("div[data-testid='review']").count()

        print(f"🔄 Clicking Load More... (current reviews: {before_count})")
        await load_more.click()

        # ✅ ждём пока количество отзывов увеличится
        await page.wait_for_function(
            "(prev) => document.querySelectorAll(\"div[data-testid='review']\").length > prev",
            arg=before_count,
            timeout=10000
        )

        await polite_sleep()

    # ✅ Теперь собираем ВСЕ отзывы
    blocks = await page.query_selector_all("div[data-testid='review']")
    print("✅ Total reviews loaded:", len(blocks))

    for b in blocks:
        date_el = await b.query_selector("span[data-testid='review-date']")
        stars_el = await b.query_selector("span[data-testid='review-stars']")
        text_el = await b.query_selector("p[data-testid='review-text']")

        date = (await date_el.inner_text()).strip() if date_el else None
        stars = 0
        if stars_el:
            svgs = await stars_el.query_selector_all("svg")
            stars = len(svgs)
        text = (await text_el.inner_text()).strip() if text_el else None

        if text:
            all_rows.append({"date": date, "stars": stars, "text": text})

    df = pd.DataFrame(all_rows)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df




async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        context = await browser.new_context(
            user_agent=USER_AGENT,
            viewport={"width": 1366, "height": 768},
            locale="en-US"
        )

        page = await context.new_page()

        products = await scrape_products(page)
        testimonials = await scrape_testimonials(page)
        reviews = await scrape_reviews(page)

        # Save
        products.to_csv("/Users/annaklimenko/Documents/dm-hw3-webscr/data/products.csv", index=False)
        testimonials.to_csv("/Users/annaklimenko/Documents/dm-hw3-webscr/data/testimonials.csv", index=False)
        reviews.to_csv("/Users/annaklimenko/Documents/dm-hw3-webscr/data/reviews.csv", index=False)

        print("\n✅ DONE!")
        print("Products:", len(products))
        print("Testimonials:", len(testimonials))
        print("Reviews:", len(reviews))

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
